module VecDiff
using LinearAlgebra
using TensorOperations
using Zygote
export input, input1, @forward, forward,forward_1,forward_2, ∇, prod322, prod32, prod32_mid, prod23,  prod21, prod31, prod22, is_zero

struct Diff2{ T <: AbstractFloat}
    data:: Array{T,1}
    derivative1:: Dict{ Diff2{T}, Array{T,2}}
    derivative2:: Dict{Tuple{Diff2{T},Diff2{T}}, Array{T,3}}
    is_first_order :: Bool
end

function wrapAsInput2(data:: Array{T,1};is_first_order=false) where T<: AbstractFloat
    s=size(data)[1]
    result=Diff2(data,Dict{Diff2{T},Array{T,2}}(),Dict{Tuple{Diff2{T},Diff2{T}}, Array{T,3}}(),is_first_order)
    result.derivative1[result]=Matrix(I,s,s)
    result
end

function wrapAsDiff2(data:: Array{T,1};is_first_order=false) where T<: AbstractFloat
    Diff2(data,Dict{Diff2{T},Array{T,2}}(),Dict{Tuple{Diff2{T},Diff2{T}}, Array{T,3}}(),is_first_order)
end

function Base.show(io::IO,x::Diff2{T}) where T<:AbstractFloat
    # the first argument is very important
    name=if(x.is_first_order) "Diff" else "Diff2" end
    print(io,"$(name)($(x.data))")
end

"""
calculate the index of array of 1d array
"""
function cal_slice(arg_data)
    s=1
    [ (l=length(data_);s=s+l;(s-l):(s-1)) for data_ in arg_data]
end

"""
intialize the result matrix  ∂f/∂x_i, for f(x1,..,x_N)
"""
function gene_initial_gradient1(arg_data,result_data)
    result_size=length(result_data)
    [zeros(result_size,length(data)) for data in arg_data]
end

"""
intialize the result matrix  ∂^2 f/∂x_i∂x_j, for f(x1,..,x_N)
"""
function gene_initial_gradient2(arg_data,result_data)
    result_size=length(result_data)
    num_inputs=length(arg_data)
    reshape([zeros(result_size,length(data2),length(data1)) for data1 in arg_data for data2 in arg_data],num_inputs,num_inputs)
end

"""
a convinient macro,
is_first_order must be defined in the running context, and expr only evaluated when it is false
"""
macro for_second_order(expr)
    esc(quote
        if(!is_first_order)
        $(expr)
        end
        end)
end

"""
compute the first order and seond order dependencies from on array of Diff2
"""
function get_dependency(args;is_first_order=false)
    inputs1=Set()
    inputs2=Set()
    for arg in args
        for key in keys(arg.derivative1)
            push!(inputs1,key)
        end
        @for_second_order for key in keys(arg.derivative2)
                push!(inputs2,key[1])
                push!(inputs2,key[2])
        end
    end    
    @for_second_order  inputs2=union(inputs1,inputs2) 
    inputs1,inputs2    
end

"""
compute is_first_order for args, the rules is, if one of the arg is is_first_order, and the result is first order. As that's the accuracy of the problem
"""
function cal_is_first_order(args)
    is_first_order=false
    for arg in args
        is_first_order=is_first_order||arg.is_first_order
    end
    is_first_order
end

"""
compute    ∂f/∂x_i, for f(x1,..,x_N)
"""
function cal_derivative1(func,arg_data,result_data)
    args_idx=cal_slice(arg_data)
    flat_arg_data=vcat(arg_data...)
    result_derivatives= gene_initial_gradient1(arg_data,result_data)
    for i in 1:length(result_data)
        merged_gradient=gradient(p->func([p[idx] for idx in args_idx]...)[i],flat_arg_data)[1]
        if(typeof(merged_gradient)!=Nothing)
            for j in 1:length(arg_data)
                result_derivatives[j][i,:]=merged_gradient[args_idx[j]]
            end
        end        
    end
    result_derivatives
end

"""
compute the  matrix  ∂^2 f/∂x_i∂x_j, for f(x1,..,x_N)
"""
function cal_derivative2(func,arg_data,result_data)
    args_idx=cal_slice(arg_data)
    flat_arg_data=vcat(arg_data...)
    result_derivatives2=gene_initial_gradient2(arg_data,result_data)
    for i in 1:length(result_data)
        merged_gradient2=Zygote.hessian(p->func([p[idx] for idx in args_idx]...)[i],flat_arg_data)
        # there are some problem when the gradeint is nothing, we may want to modify the orignal hessian function later, but right now, it should be fine for our application
        for j in 1:length(arg_data)
            for l in 1:length(arg_data)
                result_derivatives2[j,l][i,:,:]=merged_gradient2[args_idx[j],args_idx[l]]
            end
        end        
    end
    result_derivatives2
end

macro add_if_not_zero(a,b)
    esc(quote
        if($a==0)
            $a=$b
        else
            $a+=$b
        end        
    end)
end

macro assign_if_not_zero(a,b)
    esc(quote
        if($(a)!=0 && sum(abs.($(a)))>1E-14)
        $(b)=$(a)
        end        
    end)
end

"""
update the derivative1 for result
"""
function update_derivative1(result,inputs,args,result_derivatives)
    for input in inputs
        derivative=0
        for j in 1:length(result_derivatives)
            if(input in keys(args[j].derivative1))
                @add_if_not_zero derivative result_derivatives[j]*args[j].derivative1[input]
            end            
        end
        @assign_if_not_zero derivative  result.derivative1[input]
    end
end

function update_derivative2(result,inputs,args,result_derivatives,result_derivatives2)
    for input1 in inputs        
        for input2 in inputs
            derivative=0
            # there are two contribution source
            # we first consider the double coutribution, we need to define a tensor product first
            for j in 1:length(result_derivatives)
                for l in 1:length(result_derivatives)
                    if((input1 in keys(args[j].derivative1))&&(input2 in keys(args[l].derivative1)))
                        term=prod322( result_derivatives2[j,l], args[j].derivative1[input1],args[l].derivative1[input2])
                        @add_if_not_zero derivative term
                    end
                end
            end
            # Now, we consider the single contribution
            for j in 1:length(result_derivatives)
                if((input1,input2) in keys(args[j].derivative2))
                    term=prod23(result_derivatives[j],args[j].derivative2[(input1,input2)])
                    @add_if_not_zero derivative term
                end                
            end
            @assign_if_not_zero derivative result.derivative2[(input1,input2)]
        end        
    end
end


# a, 3, b,2 ,c,2
# size(rand(2,2,2))
"""
a_imn*b_mj*c_nk
"""
function prod322(a,b,c)
    d1=size(a)[1]
    d2=size(b)[2]
    d3=size(c)[2]
    result=zeros(d1,d2,d3)
    @tensor begin
        result[i,j,k]=a[i,m,n]*b[m,j]*c[n,k]
    end
    result
end

"""
a_ijm*b_mk
"""
function prod32(a,b)
    d1=size(a)[1]
    d2=size(a)[2]
    d3=size(b)[2]
    result=zeros(d1,d2,d3)
    @tensor begin
        result[i,j,k]=a[i,j,m]*b[m,k]
    end
end

"""
a_imk*b_mj
"""
function prod32_mid(a,b)
    d1=size(a)[1]
    d2=size(b)[2]
    d3=size(a)[3]
    result=zeros(d1,d2,d3)
    @tensor begin
        result[i,j,k]=a[i,m,k]*b[m,j]
    end
end

"""
a_im*b_mjk
"""
function prod23(a,b)
    d1=size(a)[1]
    d2=size(b)[2]
    d3=size(b)[3]
    result=zeros(d1,d2,d3)
    @tensor begin
        result[i,j,k]=a[i,m]*b[m,j,k]
    end
    result
end

"""
the key function to implement the forward mode
using update_derivative and update_derivative2 to customize the forward proces
update_derivative1(result,inputs1,args,result_derivatives1)
result, the result Diff, use     
result=wrapAsDiff2(result_data;is_first_order=is_first_order)
inputs, all the input nodes
args, vector of Diff
result_derivative1, result_derivative1[i]= ∂result/∂args[i] 
"""
function forward(func,args...)
    is_first_order=cal_is_first_order(args)
    arg_data=[ arg. data for arg in args]
    result_data=func(arg_data...)
    # we first merge all args together
    # the gradeint of func regarding to its arguments
    # compute derivatives to first and second order
    result_derivatives1=cal_derivative1(func,arg_data,result_data)
    @for_second_order result_derivatives2=cal_derivative2(func,arg_data,result_data)
    # get the dependencies for first order and second order derivatives
    inputs1,inputs2=get_dependency(args;is_first_order=is_first_order)
    result=wrapAsDiff2(result_data;is_first_order=is_first_order)
    update_derivative1(result,inputs1,args,result_derivatives1)
    @for_second_order update_derivative2(result,inputs2,args,result_derivatives1,result_derivatives2)
    result
end
# manuall forward
"""
result_derivative1, result_derivative1[i]= ∂result/∂args[i] 
"""
function forward_1(result_data,args,result_derivatives1)
    result=wrapAsDiff2(result_data;is_first_order=true)
    inputs1,inputs2=get_dependency(args;is_first_order=true)
    update_derivative1(result,inputs1,args,result_derivatives1)
    result
end

"""
result_derivative1, vector
result_derivative2, matrix

result_derivative1, result_derivative1[i]= ∂result/∂args[i] 
result_derivative2, result_derivative1[i]= ∂result/∂args[i] 

"""
function forward_2(result_data,args,result_derivatives1,result_derivatives2)
    result=wrapAsDiff2(result_data;is_first_order=false)
    inputs1,inputs2=get_dependency(args;is_first_order=false)
    update_derivative1(result,inputs1,args,result_derivatives1)
    update_derivative2(result,inputs1,args,result_derivatives1,result_derivatives2)
    result
end


# now, we dress the give the interface
"""
compute the first order derivative 
"""
function derivative1(a::Diff2{T},e::Diff2{T}) where T<:AbstractFloat
    if (e in keys(a.derivative1))
        return a.derivative1[e]
    end
    zeros(length(a.data),length(e.data))
end

"""
compute the second order derivative 
"""
function derivative2(a::Diff2{T},e1::Diff2{T},e2::Diff2{T}) where T<:AbstractFloat
    if ((e1,e2) in keys(a.derivative2))
        return a.derivative2[(e1,e2)]
    end
    zeros(length(a.data),length(e1.data),length(e2.data))
end


function input(x)
    wrapAsInput2(x)
end

function input1(x)
    wrapAsInput2(x;is_first_order=true)
end

# notice the esc function, which is necessary when the expression evolves local variables
macro forward(expr)
    esc(rewrite_call(expr,:(VecDiff.forward)))
end

function rewrite_call(expr,symbol)
    if(typeof(expr)==Expr)
        for arg in expr.args
            rewrite_call(arg,symbol)
        end        
        if(expr.head==:call)
            expr.args=[symbol,expr.args...]
        end
    end
    expr
end

"""
compute the first order derivative 
"""
function ∇(a::Diff2{T},e::Diff2{T}) where T <: AbstractFloat
    derivative1(a,e)
end

"""
compute the second order derivative 
"""
function ∇(a::Diff2{T},e1::Diff2{T},e2::Diff2{T}) where T <: AbstractFloat
    derivative2(a,e1,e2)
end

function mulDiffWithNumber(a::Diff2{T},b::T) where T <: AbstractFloat
    is_first_order=a.is_first_order
    result=wrapAsDiff2(a.data*b;is_first_order=is_first_order)    
    for input in keys(a.derivative1)
        result.derivative1[input]=a.derivative1[input]*b
    end
    @for_second_order for input in keys(a.derivative2)
        result.derivative2[input]=a.derivative2[input]*b
    end
    result
end


function Base.:*(a::Diff2{T},b::T) where T <: AbstractFloat
    mulDiffWithNumber(a,b)
end

function Base.:*(b::T,a::Diff2{T}) where T <: AbstractFloat
    mulDiffWithNumber(a,b)
end

function Base.:*(a::Diff2{T},b::Number) where T <: AbstractFloat
    mulDiffWithNumber(a,convert(T,b))
end

function Base.:*(b::Number,a::Diff2{T}) where T <: AbstractFloat
    mulDiffWithNumber(a,convert(T,b))
end

# this will change a, and assuming a and b are different
function addDiff!(a::Diff2{T},b::Diff2{T}) where T <: AbstractFloat
    a.data[:]=a.data+b.data
    is_first_order=cal_is_first_order([a,b])
    for input_b in keys(b.derivative1)
        if(input_b in keys(a.derivative1))
            a.derivative1[input_b]=a.derivative1[input_b]+b.derivative1[input_b]
        else
            a.derivative1[input_b]=b.derivative1[input_b]
        end     
    end
    @for_second_order for input_b in keys(b.derivative2)
        if(input_b in keys(a.derivative2))
            a.derivative2[input_b]=a.derivative2[input_b]+b.derivative2[input_b]
        else
            a.derivative2[input_b]=b.derivative2[input_b]
        end     
    end
    a
end

function subDiff!(a::Diff2{T},b::Diff2{T}) where T <: AbstractFloat
    a.data[:]=a.data-b.data
    is_first_order=cal_is_first_order([a,b])
    for input_b in keys(b.derivative1)
        if(input_b in keys(a.derivative1))
            a.derivative1[input_b]=a.derivative1[input_b]-b.derivative1[input_b]
        else
            a.derivative1[input_b]=-b.derivative1[input_b]
        end     
    end
    @for_second_order for input_b in keys(b.derivative2)
        if(input_b in keys(a.derivative2))
            a.derivative2[input_b]=a.derivative2[input_b]-b.derivative2[input_b]
        else
            a.derivative2[input_b]=-b.derivative2[input_b]
        end     
    end
    a
end

function addDiff(a::Diff2{T},b::Diff2{T}) where T <: AbstractFloat
    is_first_order=cal_is_first_order([a,b])
    result=wrapAsDiff2(zeros(size(a.data));is_first_order=is_first_order)
    addDiff!(result,a)
    addDiff!(result,b)
    result
end

function subDiff(a::Diff2{T},b::Diff2{T}) where T <: AbstractFloat
    is_first_order=cal_is_first_order([a,b])
    result=wrapAsDiff2(zeros(size(a.data));is_first_order=is_first_order)
    addDiff!(result,a)
    subDiff!(result,b)
    result
end

function getIdxDiff(a::Diff2,idx)
    is_first_order=a.is_first_order
    result=wrapAsDiff2(a.data[idx];is_first_order=is_first_order)
    for input in keys(a.derivative1)
        result.derivative1[input]=a.derivative1[input][idx,:]
    end
    @for_second_order for input in keys(a.derivative2)
        result.derivative2[input]=a.derivative2[input][idx,:,:]
    end
    result
end

function getIdxDiff(a::Diff2,idx::Integer)
    getIdxDiff(a,idx:idx)
end


function Base.:+(a::Diff2{T},b::Diff2{T}) where T <: AbstractFloat
    addDiff(a,b)
end

function Base.:-(a::Diff2{T},b::Diff2{T}) where T <: AbstractFloat
    subDiff(a,b)
end

function Base.:-(a::Diff2{T}) where T <: AbstractFloat
    a*(-1.0)
end

function Base.:+(a::Number,b::Diff2{T}) where T <: AbstractFloat
    wrapAsDiff2([convert(T,a)])+b
end

function Base.:+(b::Diff2{T},a::Number,) where T <: AbstractFloat
    wrapAsDiff2([convert(T,a)])+b
end

function Base.:-(a::Number,b::Diff2{T}) where T <: AbstractFloat
    wrapAsDiff2([convert(T,a)])-b
end

function Base.:-(b::Diff2{T},a::Number,) where T <: AbstractFloat
    b-wrapAsDiff2([convert(T,a)])
end




function Base.sum(a::Array{Diff2{T},1}) where T <: AbstractFloat
    is_first_order=cal_is_first_order(a)
    result=wrapAsDiff2(zeros(size(a[1].data));is_first_order)
    for a_ in a
        addDiff!(result,a_)
    end
    result
end

function Base.getindex(a::Diff2{T},idx) where T <: AbstractFloat
    getIdxDiff(a,idx)
end



# we also need function to cat first order and second order derivatives
function ∇(a::Diff2{T},b::Array{Diff2{T}}) where T <: AbstractFloat
    arg_data=[ arg.data for arg in b]
    idx=cal_slice(arg_data)
    result=zeros(size(a.data)[1],idx[end][end])
    for i in 1:(size(arg_data)[1])
        result[:,idx[i]]=∇(a,b[i])
    end    
    result
end


function ∇(a::Diff2{T},b::Array{Diff2{T}},c::Array{Diff2{T}}) where T <: AbstractFloat
    arg_data=[ arg.data for arg in b]
    idx_b=cal_slice(arg_data)
    arg_data=[ arg.data for arg in c]
    idx_c=cal_slice(arg_data)
    result=zeros(size(a.data)[1],idx_b[end][end],idx_c[end][end])
    for i in 1:(size(idx_b)[1])
        for j in 1:(size(idx_c)[1])
            result[:,idx_b[i],idx_c[j]]=∇(a,b[i],c[j])
        end
    end    
    result
end

"""
a_ij*b_i, i element wise
"""
function prod21(a,b)
    result=similar(a)
    w,h=size(a)
    for i in 1:w
        for j in 1:h
            result[i,j]=a[i,j]*b[i]
        end
    end
    result
end

"""
a_ijk*b_i, i element wise
"""
function prod31(a,b)
    result=similar(a)
    w,h,z=size(a)
    for i in 1:w
        for j in 1:h
            for k in 1:z
                result[i,j,k]=a[i,j,k]*b[i]
            end            
        end
    end
    result
end


"""
a_ij*b_ik, i element wise
"""
function prod22(a,b)
    w,h=size(a)
    w,z=size(b)
    result=zeros(w,h,z)
    for i in 1:w
        for j in 1:h
            for k in 1:z
                result[i,j,k]=a[i,j]*b[i,k]
            end            
        end
    end
    result
end

"""
a element-wise product for two Diff2
"""
function elem_prod(a::Diff2{T}, b::Diff2{T}) where T <: AbstractFloat
    is_first_order=cal_is_first_order([a,b])
    result=wrapAsDiff2(a.data.*b.data;is_first_order=is_first_order)
    input1,input2=get_dependency([a,b];is_first_order=is_first_order)
    for input in input1
        derivative=0
        if(input in keys(a.derivative1))
            derivative=prod21(a.derivative1[input],b.data)
        end
        if(input in keys(b.derivative1))
            term=prod21(b.derivative1[input],a.data)
            @add_if_not_zero derivative term
        end
        @assign_if_not_zero derivative result.derivative1[input]
    end
    # now, we compute the second order derivative
    @for_second_order for input_j in input2
        for input_k in input2
            # print([input_j,input_k])
            derivative=0
            if((input_j,input_k) in keys(a.derivative2))
                derivative=prod31(a.derivative2[(input_j,input_k)],b.data)
            end
            if((input_j in keys(a.derivative1) )&& (input_k in keys(b.derivative1)))
                term=prod22(a.derivative1[input_j],b.derivative1[input_k])
                @add_if_not_zero derivative term
            end
            if((input_j,input_k) in keys(b.derivative2))
                term=prod31(b.derivative2[(input_j,input_k)],a.data)
                @add_if_not_zero derivative term
            end
            if((input_k in keys(a.derivative1) )&& (input_j in keys(b.derivative1)))
                term=prod22(b.derivative1[input_j],a.derivative1[input_k])
                @add_if_not_zero derivative term
            end
            @assign_if_not_zero derivative result.derivative2[(input_j,input_k)]
        end
    end    
    result
end


function Base.:*(a::Diff2{T},b::Diff2{T}) where T <: AbstractFloat
    elem_prod(a,b)
end

function powerOverNumber(a::Diff2{T},b::Number) where T<:AbstractFloat
    is_first_order=a.is_first_order
    result=wrapAsDiff2((a.data).^b;is_first_order=is_first_order)
    inputs1,inputs2=get_dependency([a]; is_first_order=is_first_order)
    for input in inputs1
        derivative=0
        if(input in keys(a.derivative1))
            derivative=prod21(a.derivative1[input],b*((a.data).^(b-1)))
        end
        @assign_if_not_zero derivative result.derivative1[input]
    end
    @for_second_order for input_j in inputs2
        for input_k in inputs2
            derivative=0
            if(input_j in keys(a.derivative1) && input_k in keys(a.derivative1))
                derivative=prod22(prod21(a.derivative1[input_j],b*(b-1)*((a.data).^(b-2))),a.derivative1[input_k])
            end
            if((input_j,input_k) in keys(a.derivative2))
                term=prod31(a.derivative2[(input_j,input_k)],b*((a.data).^(b-1)))
                @add_if_not_zero derivative term
            end
            @assign_if_not_zero derivative result.derivative2[(input_j,input_k)]
        end
    end    
    result
end


function Base.:^(a::Diff2{T},b::Number) where T <: AbstractFloat
    powerOverNumber(a,b)
end

function vector_divide_number(a,b)
    a./b[1]
end


function Base.:/(a::Diff2{T},b::Diff2{T}) where T <: AbstractFloat
    @forward vector_divide_number(a,b)
end


#  convert array of diff into a vector, we ignore the derivatives right now
function Base.cat(a::Array{Diff2{T}}) where T <: AbstractFloat
    is_first_order=cal_is_first_order(a)
    inputs1,inputs2=get_dependency(a;is_first_order=is_first_order)
    arg_data=[ arg.data for arg in a]
    idx=cal_slice(arg_data)
    data_size=idx[end][end]
    result=wrapAsDiff2(zeros(data_size);is_first_order=is_first_order)
    # for data
    for i in 1:(size(arg_data)[1])
        result.data[idx[i]]=arg_data[i]
    end
    # for first order derivative
    for input in inputs1
        input_size=length(input.data)
        derivative=zeros(data_size,input_size)
        for i in 1:(size(arg_data)[1])
            derivative[idx[i],:]=∇(a[i],input)
        end
        result.derivative1[input]=derivative
    end
    #for second order
    @for_second_order for input_i in inputs2
        for input_j in inputs2
            input_i_size=length(input_i.data)
            input_j_size=length(input_j.data)
            derivative=zeros(data_size,input_i_size,input_j_size)
            for i in 1:(size(arg_data)[1])
                derivative[idx[i],:,:]=∇(a[i],input_i,input_j)
            end
            result.derivative2[(input_i,input_j)]=derivative
        end        
    end
    result
end

function is_zero(a::Array{T,N}) where {T<:AbstractFloat, N}
    sum(abs.(a))<1E-10
end

function is_zero(a::Diff2{T}) where T<:AbstractFloat
    if(!is_zero(a.data))
        return false
    end
    for key in keys(a.derivative1)
        if(! is_zero(a.derivative1[key]))
            return false
        end        
    end
    for key in keys(a.derivative2)
        if(! is_zero(a.derivative2[key]))
            return false
        end        
    end
    return true
end
end
