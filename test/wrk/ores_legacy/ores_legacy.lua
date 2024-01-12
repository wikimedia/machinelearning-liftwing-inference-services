-- add the parent directory to the package path in order to load the utils module
package.path = package.path .. ";../?.lua"
-- load the utils module
local utils = require("utils")

wrk.headers["User-Agent"] = "WMF ML team"
wrk.headers["Content-Type"] = "application/json"

-- initialize a thread
function setup(thread)
    return utils.setup(thread)
end

-- read an input file from command line arguments, parse it and store the data into an array of payloads
-- create a log file for each thread
function init(args)
     file = io.open(args[1], "r");
     path = {}
     for line in file:lines() do
          if (line ~= nil and line ~= '') then
              _, _, context, models, rev_ids = string.find(line, "(%w+)%s([%w|]+)%s([%w|]+)")
              table.insert(path, "/v3/scores/" .. context .. "?models=" .. models .. "&revids=" .. rev_ids );
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

-- add a random 10-50ms delay before each request
function delay()
    return utils.delay(10, 50)
end

-- change the body with different payload for each request
i = 0
function request()
     requests = requests + 1
     -- circle back to the first when used up the data
     if i == #path then
          i = 0
     end
     i = i + 1
     return wrk.format("GET", path[i], nil, nil)
end

-- process each response
function response(status, headers, body)
    return utils.response(status, headers, body)
end

-- print summary statistics for each thread
function done(summary, latency, requests)
    return utils.done(summary, latency, requests)
end
