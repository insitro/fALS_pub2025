import itertools
from typing import Dict, List, Optional, Tuple

import pandas as pd
import duckdb


def get_acquisitions(group_id: int, batch_id: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Provides dict of plate_barcodes and list of measurement_id for a given 'group_id' or 'batch_id'.

    Parameters:
    -----------
    group_id : int
        The group id
    batch_id : int, optional
        The batch id, default is None which returns all acquisitions for 'group_id'

    Returns:
    --------
     Dict[str, List[str]]
        A dictionary containing the acquisitions, where
            the keys are the plate_barcodes
            the values are list of measurement_ids
    """

    acquisitions = {
        0: {
            1: {
                "PF2618": ["b9bb0e93-dea2-4071-89a6-472a041c9ff8"],
                "PF2619": ["c168c0fc-17f0-4a55-9e3f-f4bd1c978f8d"],
                "PF2620": ["a501e92e-f89d-4c31-a272-6962d98bca24"],
                "PF2621": ["f94485c5-ec98-4c31-8fa0-331734831872"],
                "PF2622": ["daa67515-bb17-4970-933e-d82c33849011"],
                "PF2623": ["0edb2ada-9f66-4e74-943d-e72caa6458fe"],
            },
            2: {
                "PF2627": ["dac47bb3-a844-49e3-bcb5-d1f80ad0cf8b"],
                "PF2628": ["c52295f0-fd5e-4aae-be9c-ea6f450cd573"],
                "PF2629": ["686c62dc-7d9f-4191-9ee0-c2aeeae42c01"],
                "PF2630": ["5ee7a7ba-ca28-47e5-b29d-31d5506c76df"],
                "PF2632": ["d240a7e8-e6db-46db-8909-22ccceefd00b"],
                "PF2633": ["ab925112-389f-4cd5-ae03-2cd438837bba"],
            },
        },
        1: {
            1: {
                "PF2811": ["5279e7c3-36d8-4802-bc05-1ea148b6a5f7"],
                "PF2812": ["fecd2f5f-3f50-4af1-89ce-6dbed6fea572"],
                "PF2813": ["3def4583-8546-4c03-b8e1-8df696a932d8"],
                "PF2814": ["89f2ba7d-2c4e-4e58-8f9c-f7dcc0587731"],
                "PF2815": ["45a74237-3cce-4844-8a71-f52ff387f376"],
                "PF2834": ["ab4da8e7-b1d5-4525-9e97-c9293f9a1bba"],
            },
            2: {
                "PF2819": ["3e9bbf12-059e-48d9-ab38-b399a56c3d6e"],
                "PF2833": ["404bfd9b-d7ff-4447-90b0-7db937db8101"],
                "PF2821": ["09646295-b7e0-46ad-85bc-463a9a082e82"],
                "PF2822": ["7f3bca0d-99ca-4632-a741-c936d9a0c4d8"],
                "PF2823": ["e8fae690-991a-481d-8544-96e08b95bfad"],
                "PF2824": ["16b8c00c-f707-434e-9d1a-09119a86284e"],
            },
        },
        2: {
            1: {
                "PF2875": ["4f7c77a4-9be1-4592-9024-ac6753f63ab9"],
                "PF2876": ["7a99663e-381e-4a15-bfbe-e41b1d98ba4b"],
                "PF2877": ["bb17d395-5101-4890-b516-a14de68b9c51"],
                "PF2878": ["c4b129f3-43cf-454c-b341-818e376bf330"],
                "PF2879": ["e2727aaa-4428-4537-a0b2-5ac0a2e1f39d"],
                "PF2880": ["95e43db7-8bb3-4e56-95b5-452bef10c9a1"],
                "PF2883": ["c199a803-d987-45a4-ae6b-86c4d5ab9d7c"],
                "PF2884": ["913ae77f-e9bb-43bb-ac98-eb5184f08131"],
                "PF2885": ["38672be0-8c28-4720-91df-a7f3ed93c294"],
                "PF2886": ["02f2538b-301d-4568-971e-74822ca4edcb"],
                # "PF2887":["98ed0e3a-ca0d-42f9-996f-72607659fe28"], dosing map mismatched to seeding map
                "PF2888": ["dafdd99b-cfd8-43c7-8c6a-247ca483f798"],
            }
        },
        3: {
            1: {
                "PF3047": ["889baa10-b46a-4a92-a22d-7e4e12dd3e5f"],
                "PF3048": ["c9dc724f-ba39-4249-8fc7-cef109f98240"],
                "PF3049": ["c11c1375-749e-420b-84ac-52c8529c1067"],
                "PF3050": ["23a2c0eb-b43f-4e6c-b99d-acf84345009a"],
                "PF3051": ["bdba7dc3-5609-42d6-abe1-63fa971ae599"],
                "PF3052": ["59db139a-119b-419c-91f4-41760cc204e2"],
                "PF3055": ["631b637d-fac4-4ee4-a6be-a6da7d4ece05"],
                "PF3056": ["46c9528f-28f7-4fa3-9aa8-baeb0c741856"],
                "PF3057": ["f8c4f871-950f-4f1d-a111-b33ac3f14661"],
                "PF3058": ["a4fc9623-96a7-48ad-830a-4a73813aa49a"],
                "PF3059": ["3cad6f8d-79c9-4b34-82f9-ecc80240c4a1"],
                "PF3060": ["aa71caac-07bd-47a7-8fd4-b210e863a632"],
            }
        },
        4: {
            1: {
                "PF3147": ["8aa72b4e-8f99-43f6-8f1a-a6696a2173df"],
                "PF3148": ["819da5fe-db8e-4049-87b7-a77016e08c7a"],
                "PF3149": ["cfc3e893-0fdd-4c3e-8435-b05fc3cf3d81"],
                "PF3150": ["85a1df92-a8c6-4c9c-9302-330e31c65ada"],
                "PF3151": ["0015f41c-70d9-4614-a40a-a9f2f9463a7b"],
                "PF3152": ["f4219dba-0fd3-4b0c-87da-32c9d9a62636"],
                "PF3155": ["3904e73a-33f6-4dcd-924e-753060e6f956"],
                "PF3156": ["34794640-49ef-4929-aef6-a60e90164994"],
                "PF3157": ["06fd45d0-df72-4444-8156-5c682bf0eb6a"],
                "PF3158": ["83ec9000-45b0-40eb-8a94-212885ffc09b"],
                "PF3159": ["0884195c-286e-47e9-aa0f-6847bbf607a3"],
                "PF3160": ["a83d2a1a-bd5c-453a-a7da-6aa500b1742b"],
            }
        },
        # val2d10
        5: {
            1: {
                "PF3199": ["f878d32b-24a4-4b4d-a56e-83f6fe6613d6"],
                "PF3200": ["535cdee3-e7de-4250-8ff8-c76bfc5ee71a"],
                "PF3201": ["bd838f81-a885-4797-8856-db741a69e0b1"],
                "PF3202": ["a6441151-7d85-4b4c-9a2d-9f9f46ab7243"],
                "PF3203": ["dc4df10e-5540-484a-8f6a-4137777e1ecd"],
                "PF3204": ["9514553c-c95f-49da-b89a-5b6c8289186a"],
                "PF3207": ["da04f375-b093-4921-aa45-4dda7504bcd3"],
                "PF3208": ["52b03c0e-f4ed-4916-b4a6-35f6547ce6d5"],
                "PF3209": ["c90b63d4-14df-4801-873c-8d56d8c4cc96"],
                "PF3210": ["9801d625-bffb-4d1d-a46c-21f1c1d1acd0"],
                "PF3211": ["f2e03c5f-b1f3-4f36-b341-8ecb38a4cd49"],
                "PF3212": ["7f00941e-b09e-4eee-ba90-8602419c6eb6"],
            }
        },
        # validation 32 line part 1
        6: {
            1: {
                "PF3218": ["f7716a47-4b20-4f96-b340-39e4c35f203d"],
                "PF3219": ["5cb3f2eb-467a-45c8-b28e-7abf3860957a"],
                "PF3220": ["79ebff1c-3694-44f7-9997-988687f17004"],
                "PF3221": ["874cacdf-282f-4eab-9b0a-8b224477a12d"],
                "PF3222": ["7c817fd9-ec77-4c75-a200-9b7143f5c975"],
                "PF3223": ["e423093a-0201-4d1b-80ea-3e4496ec52ab"],
                "PF3226": ["7c56efb7-9a39-4469-8264-34685c507553"],
                "PF3227": ["5c2fc4ac-823a-4cde-9c2a-2a19add432ea"],
                "PF3228": ["211ca289-1ea9-4aca-94f1-18ed6abefb2a"],
                "PF3229": ["33c4ddbf-b9d1-4827-a3d4-aee1ae79964d"],
                "PF3230": ["5b2ac7d9-6823-4d08-af54-5484d70895d2"],
                "PF3231": ["8633654a-95a6-4cb0-a9dd-7fe1dea33c79"],
            }
        },
        # validation 32 line part 2
        7: {
            1: {
                "PF3244": ["5953c6c2-6426-4ce5-914b-7bb3096ad7e9"],
                "PF3245": ["39238c33-6349-41a1-b258-5c9ca71e9e18"],
                "PF3246": ["067d1299-bc5b-4f2c-ac4c-6170685d9d3d"],
                "PF3247": ["b5751d45-53ef-49c7-bbaf-1530760cc9b4"],
                "PF3250": ["71d9e605-c3b1-47ec-839d-1d80d86c42c1"],
                "PF3251": ["17ae7cca-e98b-4b84-ac18-de7440b427e8"],
                "PF3252": ["cf4cd2c5-b8b7-4fc2-9b36-3a552ceed33f"],
                "PF3253": ["ec30e91b-7e4a-4f01-ba74-330b3b574b6a"],
                "PF3254": ["fc82497b-a05a-4624-b250-1c3ad8fa45f7"],
                "PF3255": ["3913c8b4-a53a-4b37-8a7b-e14623681c3a"],
                "PF3258": ["792399bb-c38a-4b45-88df-058f47e4cb4e"],
                "PF3259": ["9d602c87-94c1-40d2-896d-777d9a3ecad0"],
            }
        },
        # validation 32 v2 line part 1
        8: {
            1: {
                "PF3276": ["684fb38a-5d33-4368-b09c-3b328ec6c8f1"],
                "PF3277": ["e909b305-a540-42d5-a6c5-1bf2dea314ac"],
                "PF3278": ["b9722c28-22ff-4fb2-a4f6-195bf9fe6f2e"],
                "PF3279": ["75995130-f50e-4902-8a57-9955ca6935c7"],
                "PF3280": ["a85ea757-da74-4f5c-8c55-c5e8c510cc2e"],
                "PF3281": ["385d9c4b-ed3f-4e73-ab03-fa5155c33128"],
                "PF3284": ["55e86eac-a0ef-420c-8dc4-e24ddb80c72a"],
                "PF3285": ["0900b56d-df8d-4843-b79a-e9d1e93c6c1b"],
                "PF3286": ["545fb64d-04f3-4b6b-826e-1690f7f89ab7"],
                "PF3287": ["7f2e51dd-a857-4a0e-bcce-b9578af1c531"],
                "PF3288": ["70e9a19e-9287-474e-8f0b-f11c1e9a2e8a"],
                "PF3289": ["d145b9b8-dc72-4812-bc8f-c31fcacc8b88"],
            }
        },
        # validation 32 v2 line part 2
        9: {
            1: {
                "PF3296": ["d318d794-f197-4ecc-ba2d-19a987e00066"],
                "PF3297": ["4f5ff4da-5420-4545-b200-a593c09c8673"],
                "PF3298": ["b18eb54b-4c02-4bbd-be55-fd61ca065660"],
                "PF3299": ["141d69b2-971c-4e6e-ba19-b4a26da4e6ac"],
                "PF3300": ["9d1c0d26-bf52-415c-b307-d8711ed35486"],
                "PF3301": ["5d1e5b6f-1d98-4e3e-9506-f837741907e7"],
                "PF3304": ["1c4f498a-b552-44df-a546-a3aac5d577b6"],
                "PF3305": ["c0c57e9a-a098-4881-a8bd-6a98650256c8"],
                "PF3306": ["0e6c8072-c60e-4037-a7e4-8ca4f3aedc12"],
                "PF3307": ["22967773-bbca-48ae-940f-1fb309167dba"],
                "PF3308": ["529d46d0-b23e-4153-9c67-c373311cc654"],
                "PF3309": ["52b79d21-90c5-4323-ae44-50f4e10f886e"],
            }
        },
        # val230705 (aka 48 lines)
        10: {
            1: {
                "PF3735": ["0551a5c4-dcc2-4413-a5d9-5dc0ef63bf4c"],
                "PF3736": ["ab27dc2f-4b68-4b8f-a450-bd598cf78f88"],
                "PF3737": ["7000d599-3954-4fc8-9e60-b9c611b409b4"],
                "PF3739": ["7ed09037-a5f9-498c-b1c2-d5eb18d0cfba"],
                "PF3740": ["134e003f-01f0-4e49-953f-ee11036c1964"],
                "PF3741": ["86aeb3cd-0a35-4c39-905c-e4d168191089"],
                "PF3743": ["fc4379f8-12b0-4b75-aec3-51c5ce565f8c"],
                "PF3744": ["89464051-5827-48c3-ab5e-6a6a81447320"],
                "PF3745": ["eb5b4c45-3e05-4d59-8f0c-ee6f88b5b828"],
            }
        },
        # val230705_r2 (aka 48 lines)
        11: {
            1: {
                "PF3838": ["364b6aee-ae09-4c82-9511-a136a94250c8"],
                "PF3842": ["f81bb288-3813-4b9d-a91e-f104af375d66"],
                "PF3843": ["dced242a-747e-4358-a1d9-4963019c89cb"],
                "PF3847": ["6480951c-f69b-44e2-9821-43dda3e76b7c"],
                "PF3848": ["21175555-66ed-4a55-9174-cee170954639"],
                "PF3836": ["ea2587dc-f49d-4245-9e0e-eb42b5f3dc2c"],
                "PF3839": ["3ba59602-d47c-4d0e-8a8d-7c7591d98550"],
                "PF3841": ["48cbfc36-93f5-427e-9122-94819b1770ae"],
                "PF3844": ["30eb22bb-f428-4d7b-b462-bc6c3a023c24"],
                "PF3849": ["c40d35f2-2f56-4958-a9d1-c2d9b3accfb8"],
                "PF3846_2": ["c1824dae-2b5d-4225-9d46-d7b13482a954"],
            }
        },
        # val230705_r3 (aka 48 lines)
        12: {
            1: {
                "PF3864": ["0beaa8dd-3e46-4fe0-8f1e-34f7ce2153b4"],
                "PF3865": ["43ca4a2d-6a10-4430-9ac9-b088a8cbec88"],
                "PF3870": ["f1a9520b-2b88-4845-928f-9e65ded08f18"],
                "PF3871": ["37eb4eb5-b4ac-41ae-9330-5e380ee5ef6f"],
                "PF3876": ["851a9e6b-1017-4962-90d9-7c03ab7e2486"],
                "PF3877": ["219b29bd-2e6b-4c02-8ea6-ae00c243823d"],
                "PF3878": ["24eb15b2-58f8-4400-8632-e4e5480b9e00"],
                "PF3879": ["65a130f9-966e-491b-9024-342eabc8fc7b"],
                "PF3867": ["8b2ca32e-474b-4595-97bf-1a3f08e1d6f9"],
                "PF3872": ["6eddb671-a73c-4371-aa8d-182e76ce6af6"],
                "PF3873": ["88c6ae54-16b3-45c9-b24c-593036915075"],
            }
        },
    }

    if not batch_id:
        result = {}
        for batch_id in acquisitions[group_id].keys():
            result.update(acquisitions[group_id][batch_id])
        return result

    return acquisitions[group_id][batch_id]


def get_group_id(plate_barcode: str, last_group: int = 5) -> int:
    """
    Determines the group ID to which a given plate barcode belongs.

    Parameters
    ----------
    plate_barcode : str
        The barcode of the plate to be assigned to a group.
    max_group : int
        The index of the last group

    Returns
    -------
    int
        The group ID to which the plate belongs, or -1 if the plate is not found in any group.
    """
    for i in range(last_group):
        if plate_barcode in get_acquisitions(group_id=i).keys():
            return i
    return -1


def get_control_dins(group_id: int, batch_id: int = None) -> str:
    """
    Get the donor's DINS of the control line for the specified group and batch.

    Parameters
    ----------
    group_id : int
        The ID of the group for which to get the control DINS.
    batch_id : int, optional
        The ID of the batch for which to get the control DINS.

    Returns
    -------
    str
        The DINS of the control line for the specified group and batch.

    Raises
    ------
    Exception
        If an unknown group_id or batch_id is specified.
    """

    if group_id == 0:
        if batch_id is None:
            raise "A batch_id must be specified for 'run_id' 1"
        elif batch_id == 1:
            return "Cins2239"
        elif batch_id == 2:
            return "Cins2022"

    return "Cins1013"


def get_disease_state(df: pd.DataFrame):
    """Determine disease state from 'disease_category'"""

    def _get_disease_state(row):
        # Need to hard code some lines since 'disease_category' metadata is missing
        patch_disease = [
            "Dins022_kiDCTN1-R1101Khom_B8",
            "Dins022_kiFUS-R521chet_G5",
            "Dins022_kiSOD1-A5Vhom_H7",
            "Dins022_koTBK1-KOhom_B5",
            "Dins023_kiDCTN1-R1101het_A11",
            "Dins023_kiFUS-R521Chet_E2",
            "Dins023_kiSOD1-A5Vhet_F7",
            "Dins023_koTBK1-KOhom_E7",
            "Dins023_koTBK1hom_E4",
            "Dins025_kiDCTN1_R1101Khom_A2",
            "Dins025_kiSOD1-A5Vhet_A8",
            "Dins032_kiDCTN1-R1101Khom_B8",
            "Dins032_kiFUS-R521Chet_H1",
            "Dins032_kiSOD1-A5Vhet_C9",
            "Dins032_kiVCP_R155Chet_E1",
            "Dins032_koTBK1hom_D5",
            "kiVCP-R155Chet_Dins033_A11",
            "kiVCP-R155Chet_Dins025_G5",
        ]

        path_non_disease = ["Dins033_mock_WT_D3"]

        if row.disease_category in [
            "Engineered Familial",
            "Familial Patient",
            "Over-Expression Familial",
        ]:
            return "disease"
        if row.disease_category in [
            "Wild-Type",
            "Engineered Familial Control",
            "Corrected Familial Mutation",
        ]:
            return "non-disease"
        if row.cell_line in patch_disease:
            return "disease"
        if row.cell_line in path_non_disease:
            return "non-disease"

        return "none"

    return df.apply(_get_disease_state, axis=1)


def get_line_pairs(group_id: int = None, batch_id: int = None) -> List[Tuple[str, str]]:
    """
    Returns a list of line pairs for the specified batch ID, or all line pairs
    if no batch ID is specified.

    Parameters
    ----------
    group_id : int
        The group ID for which to return the line pairs. If None, all line pairs
        will be returned.
    batch_id : int
        The batch ID for which to return the line pairs. If None, all line pairs
        will be returned (default is None).

    Returns
    -------
    list of tuple of str
        A list of line pairs, where each line pair is represented as a tuple of
        two donor IDs.
    """

    line_pairs = {
        0: {
            1: [
                ("Dins023_Parental_A1", "Dins023_TDP43(OE)_A1"),
                ("Dins390_corr_C9ORF72_het_C3", "Dins390_C9ORF72_A1"),
                ("Dins603_corr_C9ORF72_het_10", "Dins603_C9ORF72_A1"),
            ],
            2: [
                ("Dins022_Parental_A1", "Dins023_Parental_A1"),
                ("Dins023_Parental_A1", "Dins023_kiVCP_R155Chet_D1"),
                ("Dins604_corr_C9ORF72_het_3", "Dins604_C9ORF72_A1"),
                ("Dins605_corr_C9ORF72_het_18-2", "Dins605_C9ORF72_A1"),
            ],
        },
        1: {
            1: [
                ("Dins390_corr_C9ORF72_het_C3", "Dins390_C9ORF72_A1"),
                ("Dins603_corr_C9ORF72_het_10", "Dins603_C9ORF72_A1"),
            ],
            2: [
                ("Dins604_corr_C9ORF72_het_3", "Dins604_C9ORF72_A1"),
                ("Dins605_corr_C9ORF72_het_18-2", "Dins605_C9ORF72_A1"),
                ("Dins023_Parental_A1", "Dins023_kiVCP_R155Chet_D1"),
                ("Dins023_Parental_A1", "Dins022_Parental_A1"),
            ],
        },
        2: [
            ("Dins022_Parental_A1", "Dins022_kiTARDBP-G295Shet_D11"),
            ("Dins022_Parental_A1", "Dins022_kiTARDBP-M337Vhet_F8"),
            ("Dins022_Parental_A1", "Dins022_kiVCP-R155Chet_D7"),
            ("Dins023_Parental_A1", "Dins023_kiTARDBP-G295Shet_E3"),
            ("Dins023_Parental_A1", "Dins023_kiTARDBP-G295Shet_H7"),
            ("Dins023_Parental_A1", "Dins023_kiTARDBP-M337Vhet_D8"),
            ("Dins023_Parental_A1", "Dins023_kiTARDBP-M337Vhet_F9"),
            ("Dins032_Parental_A1", "Dins032_kiTARDBP-G295Shet_E12"),
            ("Dins032_Parental_A1", "Dins032_kiTARDBP-G295Shet_F6"),
            ("Dins032_Parental_A1", "Dins032_kiTARDBP-M337Vhet_H3"),
        ],
        3: [
            ("Dins022_mock_WT_A3", "Dins022_kiFUS-R521chet_G5"),
            ("Dins022_mock_WT_A3", "Dins022_kiSOD1-A5Vhom_H7"),
            ("Dins023_Parental_A1", "Dins023_kiFUS-R521Chet_E2"),
            ("Dins023_Parental_A1", "Dins023_kiSOD1-A5Vhet_F7"),
            ("Dins023_Parental_A1", "Dins023_mock_WT_B2"),
            ("Dins023_mock_WT_B2", "Dins023_kiFUS-R521Chet_E2"),
            ("Dins023_mock_WT_B2", "Dins023_kiSOD1-A5Vhet_F7"),
            ("Dins025_LRRK2 KI WT clone_C2", "Dins025_kiSOD1-A5Vhet_A8"),
            ("Dins032_Parental_A1", "Dins032_kiFUS-R521Chet_H1"),
            ("Dins032_Parental_A1", "Dins032_kiSOD1-A5Vhet_C9"),
            ("Dins032_Parental_A1", "Dins032_kiVCP_R155Chet_E1"),
        ],
        4: [
            ("Dins022_mock transfect WT_D2", "Dins022_kiDCTN1-R1101Khom_B8"),
            ("Dins022_mock transfect WT_D2", "Dins022_koTBK1-KOhom_B5"),
            ("Dins023_Parental_A1", "Dins023_kiDCTN1-R1101het_A11"),
            ("Dins023_Parental_A1", "Dins023_koTBK1-KOhom_E7"),
            ("Dins023_Parental_A1", "Dins023_koTBK1hom_E4"),
            ("Dins025_mock_WT_B5", "Dins025_kiDCTN1_R1101Khom_A2"),
            ("Dins032_mock_WT_B1", "Dins032_kiDCTN1-R1101Khom_B8"),
            ("Dins032_mock_WT_B1", "Dins032_koTBK1hom_D5"),
            ("Dins032_mock_WT_B1", "Dins033_mock_WT_D3"),
        ],
        5: [
            ("Dins023_Parental_A1", "Dins023_kiTARDBP-G295Shet_H7"),
            ("Dins023_mock_WT_B2", "Dins023_kiTARDBP-G295Shet_H7"),
            ("Dins023_Parental_A1", "Dins023_kiTARDBP-G295Shet_E3"),
            ("Dins023_mock_WT_B2", "Dins023_kiTARDBP-G295Shet_E3"),
            ("Dins023_Parental_A1", "Dins023_kiTARDBP-M337Vhet_F9"),
            ("Dins023_mock_WT_B2", "Dins023_kiTARDBP-M337Vhet_F9"),
            ("Dins023_Parental_A1", "Dins023_kiTARDBP-M337Vhet_D8"),
            ("Dins023_mock_WT_B2", "Dins023_kiTARDBP-M337Vhet_D8"),
            ("Dins022_mock_WT_A3", "Dins022_kiTARDBP-M337Vhet_F8"),
            ("Dins022_mock_WT_A3", "Dins022_kiTARDBP-G295Shet_D11"),
            ("Dins032_mock_WT_B1", "Dins032_kiTARDBP-G295Shet_E12"),
            ("Dins032_mock_WT_B1", "Dins032_kiTARDBP-G295Shet_F6"),
            ("Dins032_mock_WT_B1", "Dins032_kiTARDBP-M337Vhet_H3"),
        ],
        6: [
            ("Dins022_mock-WT_D2", "Dins022_mock-WT_A3"),
            ("Dins022_mock-WT_D2", "Dins022_kiSOD1-A5Vhom_H7"),
            ("Dins022_mock-WT_D2", "Dins022_kiVCP-R155Chet_D7"),
            ("Dins023_mock-WT_B2", "Dins023_Parental_A1"),
            ("Dins023_mock-WT_B2", "Dins023_kiVCP_R155Chet_D1"),
            # missing isogenic control for Dins025_kiSOD1-A5Vhet_A8
            ("Dins023_Parental_A1", "Dins025_kiSOD1-A5Vhet_A8"),
            ("Dins032_mock-WT_B1", "Dins032_kiSOD1-A5Vhet_C9"),
            ("Dins032_mock-WT_B1", "Dins032_kiVCP_R155Chet_E1"),
        ],
        7: [
            ("Dins022_mock-WT_D2", "Dins022_kiFUS-R521chet_G5"),
            ("Dins023_mock-WT_B2", "Dins023_kiFUS-R521Chet_E2"),
            ("Dins023_mock-WT_B2", "Dins023_Parental_A1"),
            ("Dins032_mock-WT_B1", "Dins032_kiFUS-R521Chet_H1"),
            ("Dins390_corr_C9ORF72_het_C3", "Dins390_C9ORF72_A1"),
            ("Dins604_corr_C9ORF72_het_3", "Dins604_C9ORF72_A1"),
            ("Dins605_corr_C9ORF72_het_18-2", "Dins605_C9ORF72_A1"),
        ],
        8: [
            ("Dins022_mock-WT_A3", "Dins022_kiSOD1-A5Vhom_H7"),
            ("Dins022_mock-WT_D2", "Dins022_kiSOD1-A5Vhom_H7"),
            ("Dins022_mock-WT_A3", "Dins022_kiVCP-R155Chet_D7"),
            ("Dins022_mock-WT_D2", "Dins022_kiVCP-R155Chet_D7"),
            ("Dins023_Parental_A1", "Dins023_kiVCP_R155Chet_D1"),
            # missing isogenic control
            ("Dins023_Parental_A1", "Dins025_kiSOD1-A5Vhet_A8"),
            ("Dins032_Parental_A1", "Dins032_mock-WT_B1"),
            ("Dins032_mock-WT_B1", "Dins032_kiSOD1-A5Vhet_C9"),
            ("Dins032_mock-WT_B1", "Dins032_kiVCP_R155Chet_E1"),
        ],
        9: [
            ("Dins022_mock-WT_D2", "Dins022_kiFUS-R521chet_G5"),
            ("Dins023_Parental_A1", "Dins023_kiFUS-R521Chet_E2"),
            ("Dins032_mock-WT_B1", "Dins032_kiFUS-R521Chet_H1"),
            ("Dins390_corr_C9ORF72_het_C3", "Dins390_C9ORF72_A1"),
            ("Dins604_corr_C9ORF72_het_3", "Dins604_C9ORF72_A1"),
            ("Dins605_corr_C9ORF72_het_18-2", "Dins605_C9ORF72_A1"),
        ],
        10: [("Dins022_mock-WT_A3", "Dins022_kiVCP-R155Chet_D7")],
        11: [("Dins022_mock-WT_A3", "Dins022_kiVCP-R155Chet_D7")],
    }

    if group_id is None:
        result = []
        for k in line_pairs.keys():
            result.extend(get_line_pairs(group_id=k))
        return result

    if not batch_id:
        if isinstance(line_pairs[group_id], List):
            return line_pairs[group_id]
        return list(itertools.chain.from_iterable(line_pairs[group_id].values()))

    return line_pairs[group_id][batch_id]


def get_gene(r: str) -> str:
    """
    Return the gene name for a given row

    Parameters
    r: str
        row from a dataframe with 'gene_group' column

    Returns
        str: gene name
    """
    if any(["C9ORF72" in x for x in r.gene_group]):
        return "C9ORF72"
    if any(["kiTARDBP-G295Shet" in r.gene_group]):
        return "kiTARDBP-G295Shet"
    if any(["kiTARDBP-M337Vhet" in r.gene_group]):
        return "kiTARDBP-M337Vhet"
    if any(["kiVCP" in x for x in r.gene_group]):
        return "kiVCP"
    if any(["kiSOD1" in x for x in r.gene_group]):
        return "kiSOD1"
    if any(["kiFUS" in x for x in r.gene_group]):
        return "kiFUS"
    if any(["kiDCTN1" in x for x in r.gene_group]):
        return "kiDCTN1"
    if any(["koTBK1" in x for x in r.gene_group]):
        return "koTBK1"


def parent_type(col):
    """
    Categorizes a column name as 'parent', 'mock', or returns the original name.

    Parameters
    ----------
    col : str
        Column name to categorize.

    Returns
    -------
    str
        Categorized column name.
    """
    if "Parent" in col or "parent" in col:
        return "parent"
    elif "mock" in col or "Mock" in col:
        return "mock"
    else:
        return col


def load_feats(feat_path):
    """
    Load features from a parquet file and preprocess the DataFrame.

    Parameters
    ----------
    feat_path : str
        Path to the parquet file containing the features.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame containing the loaded features.
    """

    df = duckdb.read_parquet(feat_path).pandas()
    df["cell_line"] = (
        df["cell_line_edit_description"]
        + "_"
        + df["donor_registry_id"]
        + "_"
        + df["cell_line_clone_or_subclone"]
    )
    df["disease_state"] = get_disease_state(df)
    return df


def load_48lines_dataset(
    feat_path, notes_path, plate_well_filters, batch_n, need_correction=False
):
    """
    Load a "48 lines" dataset from feature and notes files, apply filters, and return the resulting dataframe.

    Parameters
    ----------
    feat_path : str
        Path to the feature file.
    notes_path : str
        Path to the notes file.
    plate_well_filters : list
        List of tuples representing plate and well position filters.
    batch_n: int
        Batch number
    need_correction : bool, optional
        Flag indicating whether correction is needed. Defaults to False.

    Returns
    -------
    pd.DataFrame
        The loaded dataset after applying filters.
    """

    df_feats = load_feats(feat_path)

    if batch_n == 1:
        rename_dict = {
            "dino_als_array_run_y1kfd3n6_DAPI_TDP43_STMN2_MAP2_128crop_mask_cell_nuclei_neurite": "iDINO_DAPI+TDP43+STMN2+TUJ1",
            "dino_als_array_2ch_colorjit_rand_rot_run_7n688xib_DAPI_MAP2_128crop_mask_cell_nuclei_neurite": "iDINO_DAPI+TUJ1",
            "dino_als_array_1ch_colorjit_rand_rot_run_qoa8i8tu_DAPI_128crop_mask_nuclei": "iDINO_DAPI",
            "dino_fb_imagenet_vit_small_8_DAPI_AP2_128crop_mask_cell_nuclei_neurite": "imagenetDINO_DAPI+TUJ1",
            "dino_fb_imagenet_vit_small_8_DAPI_TDP43_STMN2_AP2_128crop_mask_cell_nuclei_neurite": "imagenetDINO_DAPI+TDP43+STMN2+TUJ1",
            "dino_als_val_4ch_patch16_100epochs_run_yp2c3igb_DAPI_TDP43_STMN2_AP2_224crop_mask_cell_nuclei_neurite": "iDINO_retrained_masked",
            "dino_als_val_4ch_patch16_100epochs_run_yp2c3igb_DAPI_TDP43_STMN2_TUJ1_224crop": "iDINO_retrained_unmasked",
            "dino_vit_small_als_jumbo_run_2ulesx6m_DAPI_TDP43_STMN2_AP2_224crop": "DINO + ViT AML (JOJO) unmasked",
            "dino_vit_small_als_jumbo_run_2ulesx6m_DAPI_TDP43_STMN2_AP2_224crop_mask_cell_nuclei_neurite": "DINO + ViT AML (JOJO) masked",
            "dino_vit_small_als_run_o0byzzni_DAPI_TDP43_STMN2_AP2_224crop_mask_cell_nuclei_neurite": "DINO + ViT AML",
        }
    elif batch_n == 2:
        rename_dict = {
            "dino_als_array_run_y1kfd3n6_DAPI_TDP43_STMN2_AP2_128crop_mask_cell_nuclei_neurite": "iDINO_DAPI+TDP43+STMN2+TUJ1",
            "dino_als_array_2ch_colorjit_rand_rot_run_7n688xib_DAPI_AP2_128crop_mask_cell_nuclei_neurite": "iDINO_DAPI+TUJ1",
            "dino_als_array_1ch_colorjit_rand_rot_run_qoa8i8tu_DAPI_128crop_mask_nuclei": "iDINO_DAPI",
            "dino_fb_imagenet_vit_small_8_DAPI_AP2_128crop_mask_cell_nuclei_neurite": "imagenetDINO_DAPI+TUJ1",
            "dino_fb_imagenet_vit_small_8_DAPI_TDP43_STMN2_AP2_128crop_mask_cell_nuclei_neurite": "imagenetDINO_DAPI+TDP43+STMN2+TUJ1",
            "dino_als_val_4ch_patch16_100epochs_run_yp2c3igb_DAPI_TDP43_STMN2_AP2_224crop_mask_cell_nuclei_neurite": "iDINO_retrained_masked",
            "dino_als_val_4ch_patch16_100epochs_run_yp2c3igb_DAPI_TDP43_STMN2_TUJ1_224crop": "iDINO_retrained_unmasked",
            "dino_vit_small_als_jumbo_run_2ulesx6m_DAPI_TDP43_STMN2_AP2_224crop": "DINO + ViT AML (JOJO) unmasked",
            "dino_vit_small_als_jumbo_run_2ulesx6m_DAPI_TDP43_STMN2_AP2_224crop_mask_cell_nuclei_neurite": "DINO + ViT AML (JOJO) masked",
            "dino_vit_small_als_run_o0byzzni_DAPI_TDP43_STMN2_AP2_224crop_mask_cell_nuclei_neurite": "DINO + ViT AML",
        }
    elif batch_n == 3:
        rename_dict = {
            "dino_als_array_1ch_colorjit_rand_rot_run_qoa8i8tu_DAPI_128crop_mask_cell_nuclei_neurite": "iDINO_DAPI",
            "dino_als_array_2ch_colorjit_rand_rot_run_7n688xib_DAPI_MAP2_128crop_mask_cell_nuclei_neurite": "iDINO_DAPI+TUJ1",
            "dino_als_array_run_y1kfd3n6_DAPI_TDP43_STMN2_MAP2_128crop_mask_cell_nuclei_neurite": "iDINO_DAPI+TDP43+STMN2+TUJ1",
        }

    df_feats.rename(rename_dict, axis=1, inplace=True)

    if need_correction:
        need_correction = df_feats[df_feats.cell_line == "corr_C9ORF_Dins522_B8"]
        corrected = df_feats[df_feats.cell_line != "corr_C9ORF_Dins522_B8"]

        need_correction["cell_line"] = "corr_C9ORF72_Dins522_B8"
        need_correction["cell_line_edit_description"] = "corr_C9ORF72"

        df_feats = pd.concat([corrected, need_correction])

    df_feats["plate_well"] = df_feats.apply(
        lambda x: x["plate_barcode"] + "-" + x["well_position"], axis=1
    )
    df_feats["plate_well"] = df_feats["plate_well"].astype(str)

    for plate_barcode, well_position in plate_well_filters:
        df_feats = df_feats[
            ~(
                (df_feats.plate_barcode == plate_barcode)
                & (df_feats.well_position == well_position)
            )
        ]

    if notes_path:
        notes = pd.read_csv(notes_path)

        # standardize colum names
        if "Plate" in notes.columns:
            plate_col = "Plate"
        else:
            plate_col = "Plate_x"
        if "Well" in notes.columns:
            notes.rename({"Well": "well_position"}, axis=1, inplace=True)

        notes["plate_well"] = notes.apply(
            lambda x: str(x[plate_col]) + "-" + str(x["well_position"]), axis=1
        )
        notes["plate_well"] = notes["plate_well"].astype(str)

        df_feats = df_feats[~df_feats.plate_well.isin(notes.plate_well.unique().astype(str))]

    df_feats["batch"] = batch_n
    df_feats["parent_type"] = df_feats["cell_line"].apply(parent_type)
    df_feats["disease_donor"] = df_feats.apply(
        lambda x: x["disease_state"] + "-" + x["donor_registry_id"], axis=1
    )

    return df_feats
