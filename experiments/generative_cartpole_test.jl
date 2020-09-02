#Replication of the generative cartpole by Yen Yu

using LinearAlgebra
using Flux
using PyCall
using Statistics, StatsBase
using Flux.Tracker: data, forward, back
import Base: abs
abs(a::AbstractArray) = abs.(a)

np = pyimport("numpy")
gym = pyimport("gym")

include("memory.jl")
env = gym.make("CartPole-v1")
s = env.reset()
a = env.action_space.sample()
env.step(0.1)
function collect_random_experience(env,mem,N)
    s = env.reset()
    for i in 1:N
        a = env.action_space.sample()
        #print(a)
        s,r,done = env.step(a)

        append(mem, Float32.(s),Float32.([a]),Float32(r),done)
        if done
            env.reset()
        end
    end
    return mem
end

const num_futures = 16
const input_size = 5
const latent_size = 2
const num_hidden = 256
const noise_std = 0.2f0
const num_recurrent_steps = 7
const learning_rate = 0.0001f0
const training_steps_per_cycle = 400
const episodes_per_cycle = 5
const num_gradient_steps = 100
const grad_descent_step_size = 0.05f0
const latent_l2_norm = 0.001f0
const concat_layer_size = 4
const l2_reg_coeff = Float32(5e-4)
const x_weight = 1.2f0
const xdot_weight = 1.8f0
const theta_weight = 3f0
const thetadot_weight = 0.8f0
const batch_size = 1000 
encoder = Chain(Dense(input_size*num_futures, num_hidden, Flux.relu),
                Dense(num_hidden, num_hidden, Flux.relu),
                Dense(num_hidden, num_hidden, Flux.relu),
                Dense(num_hidden, latent_size, Flux.tanh))

decoder = Chain(Dense(latent_size + concat_layer_size,num_hidden, Flux.relu),
                Dense(num_hidden, num_hidden, Flux.relu),
                Dense(num_hidden, num_hidden, Flux.relu),
                Dense(num_hidden, input_size * num_futures))

function model(inp, context)
        outputs = []
        output = inp
        for i in 1:num_recurrent_steps
                #println(typeof(output))
                latent = encoder(inp)
                latent += noise_std .* randn(size(latent)...)
                out = cat(latent, context, dims=1)
                output = decoder(out)
                push!(outputs, output)
        end
        return outputs
end

cat_batch(os,as) = cat(cat.(os,as,dims=1)...,dims=1) #


function reward_function(state, target_x = 0, target_theta = 0)
    R = x_weight *abs(state[1] - target_x)+ xdot_weight * state[2]^2 + theta_weight * (state[3] - target_theta) ^2 + thetadot_weight * state[4]^2
    return R
end
split_output(out) = [out[(i-1)*input_size+1:i*input_size,:] for i in 1:div(size(out,1),input_size)]
#function R(outs, target_x=0, target_theta=0) = reward_function.([outs[(i-1)*num_futures +1: i*num_futures,:] for i in 1:div(length(outs)/num_futures)], target_x, target_theta)
Rfun(outs) = sum(reward_function.(split_output(outs[end])))



ps = params(encoder, decoder)
l2(ps::Params) = sum(sum.(abs.(data.(ps)))) * l2_reg_coeff
autoencoder_loss(inp, outputs) = sum(mean([(inp .- outputs[i]).^2 for i in 1:length(outputs)]))

opt = ADAM()
##


function train(num_epochs)
    losses = []
    for i in 1:num_epochs
        println("Epoch: $i")
        os,as,rs,dones = sample_batch_naive(mem,num_futures,batch_size)
        inp = cat_batch(os,as)
        l = autoencoder_loss(inp, model(inp,os[1]))
        push!(losses, l.data)
        println("loss: $l")
        gs = Flux.Tracker.gradient(()->l, ps,nest=true)
        for p in ps
            grad = gs[p]
            println(mean(grad))
            update!(opt, p, grad)
            #if mean(grad) == 0.0
            #    println("gradients all zero")
            #    break
            #end
        end
    end
    return losses
end

const num_epochs = 1000


env.step(0)



function train(num_epochs)
    losses = []
    for i in 1:num_epochs
        println("Epoch: $i")
        os,as,rs,dones = sample_batch_naive(mem,num_futures,batch_size)
        inp = cat_batch(os,as) .+ Float32(1e-10)
        l = autoencoder_loss(inp, model(inp,os[1]))
        push!(losses, l.data)
        println("loss: $l")
        gs = Flux.Tracker.gradient(()->l, ps,nest=true)
        for p in ps
            grad = gs[p]
            #println(mean(grad))
            update!(opt, p, grad)
        end
    end
    plt = plot(losses)
    display(plt)
    return losses
end
function interact_with_env(env,mem, prev_latent = randn(2,1), targets=[0,0], render_flag=false)
    s = env.reset()
    done = false
    i = 0
    latent = copy(prev_latent)
    latent = get_policy(latent, s,targets)
    preds = decoder(cat(latent, s, dims=1)).data
    as = []
    a = get_actions(split_output(preds))[1]
    action = sample_action(a)
    total_r = 0f0
    while i <= 500 && done == false
        s,r,done,info = env.step(action)
        append(mem, Float32.(s), Float32.([a]), Float32(r),done)
        latent =get_policy(latent, s, targets)
        preds = decoder(cat(latent, s, dims=1)).data
        a = get_actions(split_output(preds))[1]
        println("a: $a")
        action = sample_action(a)
        push!(as,a)
        total_r +=Float32(r)
        i +=1
        if render_flag
            env.render()
        end
    end
    return total_r
end

function run_epoch(num_trials=20,num_train_epochs=200)
    total_rewards = []
    losses = []
    mem = ArrayMemory(50000,4,1)
    mem = collect_random_experience(env, mem, 2000)
    for trial in 1:num_trials
        ls = train(num_train_epochs)
        push!(losses, ls)
        for i in 1:5
            tot_r = interact_with_env(env,mem, randn(2,1), [0,0], true)
            push!(total_rewards, tot_r)
        end
    end
    return total_rewards, losses
end

run_epoch()
interact_with_env(env, mem)

get_actions(randn(80,1))
s = env.reset()
prev_latent = copy(randn(2,1))
cat(prev_latent,s,dims=1)
latent = get_policy()
function sample_action(a)
    r = rand()
    if r <= (a+1)/2
        return 1
    else
        return 0
    end
end

plot(losses)
os,as,rs,dones = sample_batch_naive(mem,num_futures,batch_size)
inp = cat_batch(os,as)
outs = model(inp, os[1])
autoencoder_loss(inp, outs)
os,as,rs,dones = sample_batch_naive(mem,num_futures,1)
inp = cat_batch(os,as)
l = autoencoder_loss(inp, model(inp,os[1]))
push!(losses, l.data)
println("loss: $l")
inp
os[1]
l
y,backfun = forward(()->l, ps)
backfun(1)
#a_loss(inp, outs) = sum([sum((inp .- outs[i]).^2) for i in 1:length(outs)])
#a2_loss(inp, outputs) = sum(mean([(inp .- outputs[i]).^2 for i in 1:length(outputs)]))
#a3_loss(inp, outputs) = sum(mean([(inp .- outs[i]).^2 for i in 1:length(outputs)]))
gs= Flux.Tracker.gradient(()->autoencoder_loss(inp, model(inp, os[1])),ps, nest=true)
gs.grads


W = randn(4,4)
b = randn(4,1)

x = param(randn(4,1))
f(x) =  sum(W*x + b)

y,backfun = forward(f, x)
backfun(1)

gs = gradient(()->f(x), params(x))
gs.grads
gs[x]
grad = gs[x]
x = x.data
x += grad.data
x = param(x)

gradient(() -> f(x), params(x))
gs.grads
gs[x]
gs2 = gradient(()->f(x), params(x))

gs2[x]

y = f(x)
Flux.back!(y, y.data)
y.data
gradient(y)
dump(y)
y.data
y.grad
y.tracker.grad
x.tracker.grad
function test_opt(x)
    x = param(x)
    xs = zeros(100,4)
    ys = []
    for i in 1:100
        y = f(x)
        Flux.back!(y, y.data)
        x.data .= x.data + (0.1 * x.grad) / mean(abs.(x.grad))
        push!(ys,y.data)
        xs[i,:] = x.data
    end
    return ys,xs
end

x = randn(4,1)
ys,xs = test_opt(x)
using Plots
plot(ys)
plot(xs)
ctx = randn(4)
latent = randn(2,1)
function reward_loss(ctx, latent, reward_targets)
    out = decoder(cat(ctx,latent,dims=1))
    reward = sum(reward_function.(split_output(out), targets...))
    return reward
end

function get_policy(init_latent, context, targets,num_steps = 100)
    #println("in policy: $latent")
    latent = param(copy(init_latent))
    latents = []
    Rs = []
    R = 0
    for i in 1:num_steps
        R = reward_loss(context, latent, targets)
        #println("R: $R")
        Flux.back!(R, R.data)
        #grad = latent.grad ./ mean(abs.(latent.grad))
        #latent.data .= latent.data - ((0.1 * grad_descent_step_size * grad)) #.+ (latent_l2_norm * grad))
        latent.data .= latent.data -(0.005 * latent.grad) / sum(abs.(latent.grad))
        #println("latent: $(latent.data)")
        #println("latent grad: $(latent.grad)")
        push!(Rs, copy(R.data))
        push!(latents, copy(latent.data))
    end
    println("Policy final R: $R")
    plt = plot(Rs)
    display(plt)
    return latent.data#, latents, Rs
end
interact_with_env(env,mem)
@time get_policy(lat, ctx, targets)
get_policy(randn(2,1),s,targets)
plot(Rs)
lat = randn(2,1)
ctx = randn(4)
targets = [0,0]
l, latents, Rs = get_policy(lat, ctx, targets)
plot(data.(Rs))
latent2, latents2, Rs2 = get_policy(l, ctx, targets)
plot(data.(Rs2))
latent3, latents3, Rs3 = get_policy(latent2, ctx, targets)
plot(data.(Rs3))
latent4, latents4, Rs4 = get_policy(latent3, ctx, targets)
plot(data.(Rs4))
latent4, latents4, Rs4 = get_policy(latent3, ctx, targets)
plot(data.(Rs4))
latent4, latents4, Rs4 = get_policy(latent4, ctx, targets)
plot(data.(Rs4))
Rs2 == Rs
latent4
pred = decoder(cat(latent4, ctx, dims=1)).data
as = get_actions(split_output(pred))
plot(as)
plot(data.(Rs))
pred = decoder(cat(latent, ctx, dims=1)).data
states= split_output(pred)
get_actions(states) = hcat(states...)[5,:]
plot(as)

bib = param(copy(l))
R = reward_loss(ctx, lat, targets)
lat
Rs
plot(data.(Rs))
arr = transpose(hcat(latents...))
plot(arr)
latents
mem = ArrayMemory(50000,4,1)
mem = collect_random_experience(env, mem, 2000)


function debug_train(num_epochs,mem)
    losses = []
    for i in 1:num_epochs
        println("Epoch: $i")
        os,as,rs,dones = sample_batch_naive(mem,num_futures,batch_size)
        inp = cat_batch(os,as) .+ Float32(1e-10) # just to add a tiny bit of numerical stability!s
        l,squares, mean_squares,err = debug_loss(inp, model(inp,os[1]),ps)
        if err == true
            return l, squares, mean_squares
        end
        push!(losses, l.data)
        println("loss: $l")
        gs = Flux.Tracker.gradient(()->l, ps,nest=true)
        for p in ps
            grad = gs[p]
            println("mean: $(mean(grad))")
            println("max: $(maximum(grad.data))")
            println("min: $(minimum(grad.data))")
            update!(opt, p, grad)
        end
    end
    plt = plot(losses)
    display(plt)
    return losses,squares, mean_squares
end

l, squares, mean_squares = debug_train(200,mem)
sum(mean([(inp .- outputs[i]).^2 for i in 1:length(outputs)])) + l2(ps)
function debug_loss(inp, outputs, ps)
    lreg =l2(ps)
    println("lreg: $lreg")
    squares = [(inp .- outputs[i]).^2 for i in 1:length(outputs)]
    maxi = maximum(maximum.(squares))
    mini = maximum(minimum.(squares))
    println("maxi: $maxi")
    println("mini: $mini")
    if maxi > 1e6 || mini < -1e6
        println("exceptionally large value detected!")
        return squares
    end
    #squares = mean([(inp .- outputs[i]).^2 for i in 1:length(outputs)])
    mean_squares = mean.(squares)
    println("squares: $mean_squares")
    l = sum(mean_squares) + lreg
    println("l: $l")
    if isnan(l) || isinf(l)
        println("infinity detected!")
        return l,squares, mean_squares, true
    end
    return l, squares, mean_squares, false
end

a = randn(50,50)
maximum(a)

squares
for square in squares
    println("loop")
    println(mean(square))
    println(maximum(square))
    println(minimum(square))
end

sq = squares[1]
function find_nans(arr)
    nans = []
    for i in 1:size(arr,1)
        for j in 1:size(arr,2)
            if isnan(arr[i,j])
                push!(nans, (i,j))
            end
        end
    end
    return nans
end
find_nans(squares[3])
find_nans(mem.os)
i = 0
os,as,rs,dones = sample_batch_naive(mem,num_futures,batch_size)
while !any(any.(isnan.(os)))
    global i,os,as,rs,dones
    os,as,rs,dones = sample_batch_naive(mem,num_futures,batch_size)
    i +=1
    println(i)
end
any.(isnan.(os))
o = os[end]
find_nans.(os)
@code_warntype any(isnan.(mem.os))
isnan.(mem.os)
mem.os
os
import Base.isnan
isnan(a::AbstractArray) = isnan.(a)
any(any.(isnan.(os)))
mem.length
find_nans(mem.os[:,1:mem.length])
mem.os[:,1:mem.length]
