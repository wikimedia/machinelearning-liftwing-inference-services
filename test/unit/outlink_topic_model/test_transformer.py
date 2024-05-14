import pytest
from outlink_topic_model.transformer.transformer import OutlinkTransformer


@pytest.mark.parametrize(
    "title, lang, expected",
    [
        (
            "Muumi",
            "fi",
            {
                "Q1215800",
                "Q1538681",
                "Q713750",
                "Q11883595",
                "Q146684",
                "Q55510485",
                "Q3051021",
                "Q1610914",
                "Q646164",
                "Q147538",
                "Q2635348",
                "Q498453",
                "Q5474086",
                "Q11884615",
                "Q3720480",
                "Q10503133",
                "Q201662",
                "Q2232311",
                "Q16729557",
                "Q5960585",
                "Q11883593",
                "Q5488397",
                "Q2830980",
                "Q4356095",
                "Q10690515",
                "Q248970",
                "Q3209530",
                "Q37930",
                "Q19707296",
                "Q100",
                "Q36",
                "Q2915296",
                "Q18680132",
                "Q2975898",
                "Q2337351",
                "Q11888073",
                "Q11860061",
                "Q28720714",
                "Q11857844",
                "Q11883598",
                "Q3267563",
                "Q996907",
                "Q10437676",
                "Q2914611",
                "Q5413585",
                "Q30174080",
                "Q111043919",
                "Q565",
                "Q1107",
                "Q5489942",
                "Q17383933",
                "Q623846",
                "Q3036406",
                "Q3131643",
                "Q28028702",
                "Q589198",
                "Q23039281",
                "Q11867741",
                "Q111457264",
                "Q2565509",
                "Q2659052",
                "Q4412285",
                "Q3564797",
                "Q4405481",
                "Q18688816",
                "Q30",
                "Q11850876",
                "Q11890979",
                "Q26723758",
                "Q189888",
                "Q281",
                "Q106310534",
                "Q10590159",
                "Q2604643",
                "Q62161",
                "Q111231294",
                "Q52062",
                "Q1779921",
                "Q473366",
                "Q2917633",
                "Q40840",
                "Q2386077",
                "Q2604653",
                "Q545",
                "Q120068791",
                "Q1754",
                "Q3740541",
                "Q850472",
                "Q2608368",
                "Q422902",
                "Q13233",
                "Q13569943",
                "Q2658084",
                "Q11866393",
                "Q5584974",
                "Q4354798",
                "Q19707466",
                "Q11883594",
                "Q114794840",
                "Q102071",
                "Q9027",
                "Q476764",
                "Q2762298",
                "Q2914817",
                "Q10791871",
                "Q5648683",
                "Q122100",
                "Q18693443",
                "Q19707453",
                "Q4355809",
                "Q3148689",
                "Q726673",
                "Q11896395",
                "Q14532188",
                "Q18679891",
                "Q760558",
                "Q2386048",
                "Q2914825",
                "Q272638",
                "Q1493772",
                "Q1856977",
                "Q18632511",
                "Q2893750",
                "Q6218421",
                "Q2593011",
                "Q2291325",
                "Q958494",
                "Q2917639",
                "Q2343628",
                "Q393843",
                "Q1289624",
                "Q134949",
                "Q19707459",
                "Q5398661",
                "Q2569920",
                "Q1807595",
                "Q3246478",
                "Q503862",
                "Q10672707",
                "Q2914822",
                "Q51290",
                "Q10590158",
                "Q11883591",
                "Q3739478",
                "Q11874430",
                "Q6416041",
                "Q19707447",
                "Q2521389",
                "Q11874377",
                "Q11896186",
                "Q11866938",
                "Q20379838",
                "Q1418002",
                "Q4460316",
                "Q11867838",
                "Q128449",
                "Q23045171",
                "Q11883597",
                "Q2616090",
                "Q11895348",
                "Q12252383",
                "Q5393689",
                "Q3505636",
                "Q11894139",
                "Q2915808",
                "Q18352682",
                "Q1546521",
                "Q745",
                "Q839492",
                "Q17",
                "Q11892368",
                "Q4405454",
                "Q145",
                "Q1412",
                "Q11883596",
                "Q2624126",
                "Q621213",
                "Q11861884",
                "Q2503697",
                "Q3744199",
                "Q185583",
                "Q11856982",
                "Q7070758",
                "Q11875977",
                "Q2609816",
                "Q945793",
                "Q7535",
                "Q6162970",
                "Q3735893",
                "Q2746094",
                "Q2601227",
                "Q926046",
                "Q4983949",
                "Q44319298",
                "Q258313",
                "Q159",
                "Q10561778",
                "Q2914480",
                "Q2918370",
                "Q201821",
                "Q4346355",
                "Q4951118",
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_outlinks(title, lang, expected):
    t = OutlinkTransformer("model", "predictor_host")
    qids = await t.get_outlinks(title, lang)
    assert qids == expected
