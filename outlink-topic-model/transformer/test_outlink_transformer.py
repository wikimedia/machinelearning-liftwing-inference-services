import pytest

from outlink_transformer.outlink_transformer import get_outlinks


@pytest.mark.parametrize(
    "title, lang, expected",
    [
        (
            "Toni Morrison",
            "en",
            {
                "Q230650",
                "Q41694060",
                "Q1242948",
                "Q4679309",
                "Q4749276",
                "Q4656363",
                "Q743777",
                "Q4763360",
                "Q278543",
                "Q275600",
                "Q705816",
                "Q47484",
                "Q216070",
                "Q167477",
                "Q4737563",
                "Q4657131",
                "Q233762",
                "Q38351",
                "Q354783",
                "Q457601",
                "Q384121",
                "Q4735950",
                "Q733850",
                "Q4733568",
                "Q442759",
                "Q4765305",
                "Q230935",
                "Q168269",
                "Q239135",
                "Q4689674",
                "Q235899",
                "Q3246738",
                "Q4626587",
                "Q234819",
                "Q4705004",
                "Q4708115",
                "Q4807200",
                "Q446574",
                "Q215868",
                "Q17098771",
                "Q325945",
                "Q627749",
                "Q4681995",
                "Q9062647",
                "Q229840",
                "Q466089",
                "Q1654349",
                "Q4646796",
                "Q4725707",
                "Q272638",
                "Q403146",
                "Q3355",
                "Q3656266",
                "Q1978467",
                "Q3784220",
                "Q191472",
                "Q215410",
                "Q2842807",
                "Q4767229",
                "Q2754186",
                "Q532873",
                "Q4686819",
                "Q213300",
                "Q42443",
                "Q4766433",
                "Q463319",
                "Q49325",
                "Q2820848",
                "Q45578",
                "Q4754260",
                "Q317877",
                "Q125121",
                "Q4767213",
                "Q445912",
                "Q231983",
                "Q463281",
                "Q1765120",
                "Q155712",
                "Q549506",
                "Q4184",
                "Q235615",
                "Q160456",
                "Q4732761",
                "Q764812",
                "Q18109559",
                "Q4732733",
                "Q4769295",
                "Q53550",
                "Q434513",
                "Q76",
                "Q463606",
                "Q89128017",
                "Q4762467",
                "Q127328",
                "Q261056",
                "Q470057",
                "Q4769275",
                "Q4706371",
                "Q4725769",
                "Q377490",
                "Q3363072",
                "Q4732508",
                "Q601307",
                "Q40469",
                "Q206191",
                "Q34670",
                "Q2849426",
                "Q14686969",
                "Q388170",
                "Q4726082",
                "Q463271",
                "Q4742229",
                "Q11608",
                "Q1533003",
                "Q15439592",
                "Q4659903",
                "Q34474",
                "Q2842834",
                "Q4546133",
                "Q4726071",
                "Q1431868",
                "Q4669153",
                "Q4800412",
                "Q4858812",
                "Q443096",
                "Q4655421",
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_outlinks(title, lang, expected):
    outlink_qids = await get_outlinks(title, lang, 100)
    assert outlink_qids == expected