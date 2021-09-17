# in PKg mode, dev VecDiff
using Revise

using VecDiff
# create some value
x_=rand(10)
# track the dependency of x up to first order
function test_func(a,b)
    [sum(a)*sum(b.^2),sum(b)*sum(a.^2)]
end

x=input1(x_)
y=x*2.0
z=@track test_func(x,y)

∂y∂x=∇(y,x)
∂y∂x∂x=∇(y,x,x)
∇(z,x)
∇(z,x,x)


x=input1(x_)
y=x*2.0
# z=@track test_func(x,y) 
z=@forward test_func(x,y) 

∂y∂x=∇(y,x)
∂y∂x∂x=∇(y,x,x)
∇(z,x)
∇(z,x,x)


# now, we add customized forward approach and test it
x=input1(rand(10))
y=input1(rand(5))
z1=forward(test_func,x,y)
z_data=z1.data
args=[x,y]
derivatives=[∇(z1,arg_) for arg_ in args]
z2=forward_1(z_data,args,derivatives)
is_zero(z1-z2)

x=input(rand(10))
y=input(rand(5))
z1=forward(test_func,x,y)
# or
z1=@forward test_func(x,y)
z_data=z1.data
args=[x,y]
derivatives=[∇(z1,arg_) for arg_ in args]
# reshape([(i,j) for j in 1:4 for i in 1:4],4,4) 
# notice the order, as i will increase first, and agress with row major
derivatives2=reshape([∇(z1,arg1,arg2) for arg2 in args for arg1 in args],length(args),length(args))
z2=forward_2(z_data,args,derivatives,derivatives2)
is_zero(z1-z2)
# ∇(z2,x,y)
# ∇(z2,y,x)


