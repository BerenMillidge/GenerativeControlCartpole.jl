import StatsBase: sample
using BSON

abstract type Memory end

mutable struct ArrayMemory <: Memory
    os::Matrix{Float32}
    as::Matrix{Float32}
    rs::Vector{Float32}
    dones::Array{Bool,1}
    length::Int
    max_size::Int
end
ArrayMemory(max_size, observation_size, action_size) = ArrayMemory(Matrix{Float32}(undef, observation_size, max_size), Matrix{Float32}(undef,action_size, max_size), Vector{Float32}(undef, max_size),Array{Bool}(undef, max_size), 0,max_size)

function sample(m::ArrayMemory, len_sample::Integer)
    idx = Int(round(rand() * (m.length - len_sample))) + 1
    return m.os[:,idx:idx+(len_sample-1)],m.as[:,idx:idx+(len_sample-1)],m.rs[idx:idx+(len_sample-1)],m.dones[idx:idx+(len_sample-1)]
end
function sample_batch_naive(m::ArrayMemory, len_sample::Int, num_batch::Int)
    o_batch = [zeros(Float32,size(m.os,1), num_batch) for i in 1:len_sample]
    a_batch = [zeros(Float32,size(m.as,1), num_batch) for i in 1:len_sample]
    r_batch = [zeros(Float32,num_batch) for i in 1:len_sample]
    done_batch = [zeros(Bool,num_batch) for i in 1:len_sample]
    for i in 1:num_batch
        os, as, rs, dones = sample(m, len_sample)
        for j in 1:len_sample
            o_batch[j][:,i] = os[:,j]
            a_batch[j][:,i] = as[:,j]
            r_batch[j][i] = rs[j]
            done_batch[j][i] = dones[j]
        end
    end
    return o_batch, a_batch, r_batch, done_batch
end

function append(m::ArrayMemory, s,a,r,done)
    l = m.length +1
    if m.length >= m.max_size
        #print(length)
        l = m.length % m.max_size
    end
    #print(l)
    m.os[:,l] = s
    m.as[:,l] = a
    m.rs[l] = r
    m.dones[l] = done
    m.length+=1
end
