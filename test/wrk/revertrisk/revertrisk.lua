-- add the parent directory to the package path in order to load the utils module
package.path = package.path .. ";../?.lua"
-- load the utils module
local utils = require("utils")

-- set HTTP method and headers for requests
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

-- initialize a thread
function setup(thread)
    return utils.setup(thread)
end

-- read an input file from command line arguments, parse it and store the data into an array of payloads
-- create a log file for each thread
function init(args)
    file = io.open(args[1], "r");
    data = {}
    for line in file:lines() do
        if (line ~= nil and line ~= '') then
            _, _, lang, rev_id = string.find(line, "(%a+)%s+(%d+)")
            table.insert(data, '{"lang": "' .. lang .. '", "rev_id": ' .. rev_id ..'}');
        end
    end
    file:close();
    local filename = "wrk_%d.log"
    logfile = io.open(filename:format(id), "w");
    local msg = "thread %d created logfile wrk_%d.log created"
    print(msg:format(id, id))
end

-- add a random 10-50ms delay before each request
function delay()
    return utils.delay(10, 50)
end

-- change the body with different payload for each request
function request()
    return utils.request(data)
end

-- process each response
function response(status, headers, body)
    return utils.response(status, headers, body)
end

-- print summary statistics for each thread
function done(summary, latency, requests)
    return utils.done(summary, latency, requests)
end
