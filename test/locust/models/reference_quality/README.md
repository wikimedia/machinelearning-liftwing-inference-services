# Load testing result
## sample_all_en_fr_ru.input

| Type       | Name                              | # reqs | # fails   | Avg  | Min | Max  | Med | req/s | failures/s |
|------------|-----------------------------------|--------|-----------|------|-----|------|-----|-------|------------|
| POST       | /v1/models/reference-need:predict | 33     | 0 (0.00%) | 412  | 172 | 1150 | 330 |  0.28 | 0.00       |
| POST       | /v1/models/reference-risk:predict | 35     | 0 (0.00%) | 169  | 135 |  429 | 150 |  0.29 | 0.00       |
| Aggregated |                                   | 68     | 0 (0.00%) | 287  | 135 | 1150 | 210 |  0.57 | 0.00       |

### Response time percentiles (approximated)

| Type       | Name                              | 50% | 66%  | 75%  | 80%  | 90%  | 95%  | 98%  | 99%  | 99.9% | 99.99% | 100% | # reqs |
|------------|-----------------------------------|-----|------|------|------|------|------|------|------|-------|--------|------|--------|
| POST       | /v1/models/reference-need:predict | 330 | 400  |  500 | 500  |  800 | 1100 | 1200 | 1200 |  1200 |  1200  | 1200 |    33  |
| POST       | /v1/models/reference-risk:predict | 150 | 160  |  160 | 180  |  190 |  400 |  430 |  430 |   430 |   430  |  430 |    35  |
| Aggregated |                                   | 210 | 270  |  350 | 400  |  500 |  800 | 1100 | 1200 |  1200 |  1200  | 1200 |    68  |