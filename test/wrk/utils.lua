-- utils.lua: a module with shared functions used to run wrk load tests for LiftWing isvcs.
-- see article_descriptions.lua for an example.
-- wrk framework scripting docs: https://github.com/wg/wrk/blob/master/SCRIPTING


local _M = {}

-- table to store thread objects
threads = {}
-- counter to assign unique IDs to each thread
counter = 1
-- counters for requests and responses (across all threads)
requests = 0
responses = 0

-- setup function that is called by each thread created by wrk
function _M.setup(thread)
    thread:set("id", counter)
    table.insert(threads, thread)
    counter = counter + 1
end

-- add a random delay between min and max milliseconds before each request
function _M.delay(min, max)
    return math.random(min, max)
end

-- change the body with different payload for each request
_M.i = 0
function _M.request(data)
    requests = requests + 1
    if _M.i == #data then
        _M.i = 0
    end
    _M.i = _M.i + 1
    return wrk.format(nil, nil, nil, data[_M.i])
end

-- process each response
function _M.response(status, headers, body)
    responses = responses + 1
    logfile:write("status:" .. status .. "\n" .. body .. "\n-------------------------------------------------\n")
end

-- print summary statistics for each thread
function _M.done(summary, latency, requests)
    for index, thread in ipairs(threads) do
        local id        = thread:get("id")
        local requests  = thread:get("requests")
        local responses = thread:get("responses")
        local msg = "thread %d made %d requests and got %d responses"
        print(msg:format(id, requests, responses))
    end
end

-- return the module table
return _M
