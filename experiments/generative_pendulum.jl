using LinearAlgebra
using Flux
using PyCall
using Statistics, StatsBase
using Plots
using Flux.Tracker: data, forward, back,Params, update!
import Base: abs

np = pyimport("numpy")
gym = pyimport("gym")

include("cartpole_memory.jl")
env = gym.make("Pendulum-v0")
s = env.reset()
acos(s[1])
asin(s[2])
a = env.action_space.sample()
Float32.(a)

function collect_random_experience(env,mem,N)
    s = env.reset()
    for i in 1:N
        a = env.action_space.sample()
        s,r,done = env.step(a)
        append(mem, Float32.(s),Float32.(a),Float32(r),done)
        if done
            env.reset()
        end
    end
    return mem
end
const num_futures = 16
const observation_size = 3
const action_size = 1
const input_size = observation_size + action_size
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
const concat_layer_size = observation_size
const l2_reg_coeff = Float32(5e-4)
const x_weight = 1.2f0
const xdot_weight = 1.8f0
const theta_weight = 3f0
const thetadot_weight = 0.8f0
const batch_size = 50

encoder = Chain(Dense(input_size*num_futures, num_hidden, Flux.relu),
                Dense(num_hidden, num_hidden, Flux.relu),
                Dense(num_hidden, num_hidden, Flux.relu),
                Dense(num_hidden, latent_size, Flux.tanh))

decoder = Chain(Dense(latent_size + concat_layer_size,num_hidden, Flux.relu),
                Dense(num_hidden, num_hidden, Flux.relu),
                Dense(num_hidden, num_hidden, Flux.relu),
                Dense(num_hidden, input_size * num_futures))

mult(arr, scalar) = [arr[i] .* scalar for i in 1:length(arr)]

function scale_preds(preds)
    scale = [1,1,8,2] # thetadot can go from -8 to 8 and a from -2 to 2
    scaled = repeat(scale, outer=[num_futures, batch_size])
    return tanh.(preds) .* scaled
end

function model(inp, context)
        outputs = []
        output = inp
        scale = [1,1,8,2] # thetadot can go from -8 to 8 and a from -2 to 2
        scaled = repeat(scale, outer=[num_futures, batch_size])
        for i in 1:num_recurrent_steps
                latent = encoder(inp)
                latent += noise_std .* randn(size(latent)...)
                out = cat(latent, context, dims=1)
                output = decoder(out)
            
                output = tanh.(output)  # tanh to push from -1 to 1, scaled to scale
                push!(outputs, output)
        end
        return mult(outputs, scaled) 
end
##
abs(a::AbstractArray) = abs.(a)
get_actions(states) = hcat(states...)[observation_size+1,:]
cat_batch(os,as) = cat(cat.(os,as,dims=1)...,dims=1)
#reward_function(state, target_x = 0, target_theta = 0) = x_weight *abs(state[1] - target_x)+ xdot_weight * state[2]^2 + theta_weight * (state[3] - target_theta) ^2 + thetadot_weight * state[4]^2
reward_function(state, targ1=0, targ2 =0) = -(acos(state[1])^2 + 0.1*state[3] + 0.001 * state[4]^2)
split_output(out) = [out[(i-1)*input_size+1:i*input_size,:] for i in 1:div(size(out,1),input_size)]
Rfun(outs) = sum(reward_function.(split_output(outs[end])))
ps = params(encoder, decoder)
l2(ps::Params) = sum(sum.(abs.(data.(ps)))) * l2_reg_coeff
autoencoder_loss(inp, outputs,ps) = mean([sum((inp .- outputs[i]).^2)  for i in 1:length(outputs)]) + l2(ps)
opt = ADAM(learning_rate)
##

function reward_loss(ctx, latent, reward_targets)
    out = scale_preds(decoder(cat(ctx,latent,dims=1)))
    reward = sum(reward_function.(split_output(out), reward_targets...))
    return reward
end
function get_policy(init_latent, context,num_steps = 100)
    latent = param(copy(init_latent))
    latents = []
    Rs = []
    R = 0
    for i in 1:num_steps
        R = reward_loss(context, latent, [0,0])
        #println("R: $R")
        Flux.back!(R, R.data)
        latent.data .= latent.data -(0.005 * latent.grad) / sum(abs.(latent.grad))
        #println("latent: $(latent.data)")
        #println("latent grad: $(latent.grad)")
        push!(Rs, copy(R.data))
        push!(latents, copy(latent.data))
    end
    println("Policy final R: $R")
    plt = plot(Rs)
    display(plt)
    return latent.data
end

function train(num_epochs,mem)
    losses = []
    for i in 1:num_epochs
        println("Epoch: $i")
        os,as,rs,dones = sample_batch_naive(mem,num_futures,batch_size)
        inp = cat_batch(os,as)
        l = autoencoder_loss(inp, model(inp,os[1]),ps)
        push!(losses, l.data)
        println("loss: $l")
        gs = Flux.Tracker.gradient(()->l, ps,nest=true)
        for p in ps
            grad = gs[p]
            update!(opt, p, grad)
        end
    end
    plt = plot(losses)
    display(plt)
    return losses
end
function interact_with_env(env,mem, prev_latent, render_flag)
    s = env.reset()
    done = false
    i = 0
    latent = copy(prev_latent)
    latent = get_policy(latent, s)
    preds = scale_preds(decoder(cat(latent, s, dims=1)).data)
    as = []
    a = get_actions(split_output(preds))[1]
    action = a
    total_r = 0f0
    while i <= 500 && done == false
        println("a: $a, action: $action")

        s,r,done,info = env.step([action])
        append(mem, Float32.(s), Float32.([a]), Float32(r),done)
        latent =get_policy(latent, s)
        preds = scale_preds(decoder(cat(latent, s, dims=1)).data)
        a = get_actions(split_output(preds))[1]
        action = a
        println("a: $a, action: $action")
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
    mem = ArrayMemory(50000,observation_size, action_size)
    mem = collect_random_experience(env, mem, 2000)
    for trial in 1:num_trials
        ls = train(num_train_epochs,mem)
        push!(losses, ls)
        for i in 1:5
            tot_r = interact_with_env(env,mem, randn(2,1), true)
            push!(total_rewards, tot_r)
        end
    end
    return total_rewards, losses
end
##
total_rewards = []
losses = []
mem = ArrayMemory(50000,observation_size, action_size)
mem = collect_random_experience(env, mem, 2000)
os,as,rs,dones = sample_batch_naive(mem,num_futures,batch_size)
inp = cat_batch(os,as)
model(inp, os[1])


latent = encoder(inp)
latent += noise_std .* randn(size(latent)...)
out = cat(latent, os[1], dims=1)
output = decoder(out)
idxs(i) = collect(range(1,64, step=4))
tanh.(output)
a = [1,1,8,2]
b = repeat(a, outer=[16,50])

out = tanh.(output)
scale = [1,1,8,2]
scale = repeat(a, outer=[num_futures, batch_size])
out .* scale
output = decoder(out)
push!(outputs, output)
l = autoencoder_loss(inp, model(inp,os[1]),ps)


total_rewards, losses = run_epoch()
plot(total_rewards)
for l in losses
    plt = plot(l)
    display(plt)
    sleep(1)
end


plot(total_rewards)
