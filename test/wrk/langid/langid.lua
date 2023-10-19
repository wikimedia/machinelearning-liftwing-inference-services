wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.headers["Host"] = "langid.llm.wikimedia.org"


-- give each thread an id
local counter = 1
local threads = {}

function setup(thread)
     thread:set("id", counter)
     table.insert(threads, thread)
     counter = counter + 1
end

-- read an input file from command line arguments, parse it and store the data into an array of payloads
-- create a log file for each thread
function init(args)
     file = io.open(args[1], "r")
     data = {}
     if not file then
        print("Error: Could not open file " .. filename)
        return
     end

     for line in file:lines() do
         if (line ~= nil and line ~= '') then
             table.insert(data, '{"text": "' .. line ..'"}');
         end
     end

     file:close();
     requests  = 0
     responses = 0
     local filename = "wrk_%d.log"
     logfile = io.open(filename:format(id), "w");
     local msg = "thread %d created logfile wrk_%d.log created"
     print(msg:format(id, id))
end

-- (optional) add a random 10-50ms delay before each request
function delay()
     return math.random(10, 50)
end

-- change the body with different payload for each request
i = 0
function request()
     requests = requests + 1
     -- circle back to the first when used up the data
     if i == #data then
          i = 0
     end
     i = i + 1
     return wrk.format(nil, nil, nil, data[i])
end

function response(status, headers, body)
     responses = responses + 1
     logfile:write("status:" .. status .. "\n" .. body .. "\n-------------------------------------------------\n");
end

function done(summary, latency, requests)
     for index, thread in ipairs(threads) do
        local id        = thread:get("id")
        local requests  = thread:get("requests")
        local responses = thread:get("responses")
        local msg = "thread %d made %d requests and got %d responses"
        print(msg:format(id, requests, responses))
     end
end
