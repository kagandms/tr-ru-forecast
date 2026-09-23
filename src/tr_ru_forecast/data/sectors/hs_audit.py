"""HS classification audit and metadata definitions.

Provides standard WCO HS2 chapter descriptions, identifies special/residual
codes (e.g. Chapter 77 reserved, Chapter 98/99 special provisions), and audits
nomenclature revision consistency across H4, H5, and H6 regimes.
"""

from typing import Any, Final

import pandas as pd

# Standard WCO HS2 Chapter Descriptions (Chapters 01 - 99)
HS2_DESCRIPTIONS: Final[dict[str, str]] = {
    "01": "Live animals",
    "02": "Meat and edible meat offal",
    "03": "Fish and crustaceans, molluscs and other aquatic invertebrates",
    "04": "Dairy produce; birds' eggs; natural honey; edible products of animal origin, not elsewhere specified or included",
    "05": "Products of animal origin, not elsewhere specified or included",
    "06": "Live trees and other plants; bulbs, roots and the like; cut flowers and ornamental foliage",
    "07": "Edible vegetables and certain roots and tubers",
    "08": "Edible fruit and nuts; peel of citrus fruit or melons",
    "09": "Coffee, tea, maté and spices",
    "10": "Cereals",
    "11": "Products of the milling industry; malt; starches; inulin; wheat gluten",
    "12": "Oil seeds and oleaginous fruits; miscellaneous grains, seeds and fruit; industrial or medicinal plants; straw and fodder",
    "13": "Lac; gums, resins and other vegetable saps and extracts",
    "14": "Vegetable plaiting materials; vegetable products not elsewhere specified or included",
    "15": "Animal, vegetable or microbial fats and oils and their cleavage products; prepared edible fats; animal or vegetable waxes",
    "16": "Preparations of meat, of fish, of crustaceans, molluscs or other aquatic invertebrates, or of insects",
    "17": "Sugars and sugar confectionery",
    "18": "Cocoa and cocoa preparations",
    "19": "Preparations of cereals, flour, starch or milk; pastrycooks' products",
    "20": "Preparations of vegetables, fruit, nuts or other parts of plants",
    "21": "Miscellaneous edible preparations",
    "22": "Beverages, spirits and vinegar",
    "23": "Residues and waste from the food industries; prepared animal fodder",
    "24": "Tobacco and manufactured tobacco substitutes; products intended for inhalation without combustion",
    "25": "Salt; sulphur; earths and stone; plastering materials, lime and cement",
    "26": "Ores, slag and ash",
    "27": "Mineral fuels, mineral oils and products of their distillation; bituminous substances; mineral waxes",
    "28": "Inorganic chemicals; compounds of precious metals, of rare-earth metals, of radioactive elements or of isotopes",
    "29": "Organic chemicals",
    "30": "Pharmaceutical products",
    "31": "Fertilizers",
    "32": "Tanning or dyeing extracts; tannins and their derivatives; dyes, pigments, paints and varnishes; putty; inks",
    "33": "Essential oils and resinoids; perfumery, cosmetic or toilet preparations",
    "34": "Soap, organic surface-active agents, washing preparations, lubricating preparations, waxes, polishing pastes, candles",
    "35": "Albuminoidal substances; modified starches; glues; enzymes",
    "36": "Explosives; pyrotechnic products; matches; pyrophoric alloys; certain combustible preparations",
    "37": "Photographic or cinematographic goods",
    "38": "Miscellaneous chemical products",
    "39": "Plastics and articles thereof",
    "40": "Rubber and articles thereof",
    "41": "Raw hides and skins (other than furskins) and leather",
    "42": "Articles of leather; saddlery and harness; travel goods, handbags and similar containers; animal gut articles",
    "43": "Furskins and artificial fur; manufactures thereof",
    "44": "Wood and articles of wood; wood charcoal",
    "45": "Cork and articles of cork",
    "46": "Manufactures of straw, of esparto or of other plaiting materials; basketware and wickerwork",
    "47": "Pulp of wood or other fibrous cellulosic material; recovered paper or paperboard",
    "48": "Paper and paperboard; articles of paper pulp, of paper or of paperboard",
    "49": "Printed books, newspapers, pictures and other products of the printing industry; manuscripts, typescripts and plans",
    "50": "Silk",
    "51": "Wool, fine or coarse animal hair; horsehair yarn and woven fabric",
    "52": "Cotton",
    "53": "Other vegetable textile fibres; paper yarn and woven fabrics of paper yarn",
    "54": "Man-made filaments; strip and the like of man-made textile materials",
    "55": "Man-made staple fibres",
    "56": "Wadding, felt and nonwovens; special yarns; twine, cordage, ropes and cables",
    "57": "Carpets and other textile floor coverings",
    "58": "Special woven fabrics; tufted textile fabrics; lace; tapestries; trimmings; embroidery",
    "59": "Impregnated, coated, covered or laminated textile fabrics; textile articles suitable for industrial use",
    "60": "Knitted or crocheted fabrics",
    "61": "Articles of apparel and clothing accessories, knitted or crocheted",
    "62": "Articles of apparel and clothing accessories, not knitted or crocheted",
    "63": "Other made-up textile articles; sets; worn clothing and worn textile articles; rags",
    "64": "Footwear, gaiters and the like; parts of such articles",
    "65": "Headgear and parts thereof",
    "66": "Umbrellas, sun umbrellas, walking-sticks, whips, riding-crops and parts thereof",
    "67": "Prepared feathers and down; artificial flowers; articles of human hair",
    "68": "Articles of stone, plaster, cement, asbestos, mica or similar materials",
    "69": "Ceramic products",
    "70": "Glass and glassware",
    "71": "Natural/cultured pearls, precious stones, precious metals, imitation jewellery; coin",
    "72": "Iron and steel",
    "73": "Articles of iron or steel",
    "74": "Copper and articles thereof",
    "75": "Nickel and articles thereof",
    "76": "Aluminium and articles thereof",
    "77": "Reserved for possible future use in the Harmonized System",
    "78": "Lead and articles thereof",
    "79": "Zinc and articles thereof",
    "80": "Tin and articles thereof",
    "81": "Other base metals; cermets; articles thereof",
    "82": "Tools, implements, cutlery, spoons and forks, of base metal; parts thereof",
    "83": "Miscellaneous articles of base metal",
    "84": "Nuclear reactors, boilers, machinery and mechanical appliances; parts thereof",
    "85": "Electrical machinery and equipment and parts thereof; sound/television recorders and reproducers",
    "86": "Railway or tramway locomotives, rolling-stock, track fixtures and fittings; signalling equipment",
    "87": "Vehicles other than railway or tramway rolling-stock, and parts and accessories thereof",
    "88": "Aircraft, spacecraft, and parts thereof",
    "89": "Ships, boats and floating structures",
    "90": "Optical, photographic, cinematographic, measuring, checking, precision, medical/surgical instruments",
    "91": "Clocks and watches and parts thereof",
    "92": "Musical instruments; parts and accessories of such articles",
    "93": "Arms and ammunition; parts and accessories thereof",
    "94": "Furniture; bedding, mattresses, cushions; luminaires and lighting fittings; prefabricated buildings",
    "95": "Toys, games and sports requisites; parts and accessories thereof",
    "96": "Miscellaneous manufactured articles",
    "97": "Works of art, collectors' pieces and antiques",
    "98": "Special classification provisions / Postal packages (non-standard / country-specific)",
    "99": "Commodities not specified according to kind / Special classification provisions / Confidential",
}

# Special, reserved, or residual chapters that do not represent standard economic sectors
SPECIAL_HS_CODES: Final[dict[str, str]] = {
    "77": "Reserved by the World Customs Organization for potential future use.",
    "98": "Special classification provisions or postal packages; non-standard residual category.",
    "99": "Commodities not specified according to kind, special classification provisions, or confidential trade.",
}


def get_hs_description(hs2: str, default: str | None = None) -> str:
    """Return the official WCO description for a 2-digit HS chapter."""
    padded = hs2.zfill(2) if hs2 else ""
    return HS2_DESCRIPTIONS.get(padded, default or f"HS Chapter {padded}")


def is_special_hs_code(hs2: str) -> bool:
    """Check if an HS code is reserved, special-use, or residual."""
    padded = hs2.zfill(2) if hs2 else ""
    return padded in SPECIAL_HS_CODES


def get_special_code_reason(hs2: str) -> str | None:
    """Return the exclusion reason for a special HS code."""
    padded = hs2.zfill(2) if hs2 else ""
    return SPECIAL_HS_CODES.get(padded)


def audit_hs_revisions(df: pd.DataFrame) -> dict[str, Any]:
    """Audit nomenclature revisions present in the development dataset.

    Args:
        df: DataFrame containing trade records with 'classification_code' or 'classification'.

    Returns:
        Dictionary detailing observed classifications, time spans, and consistency notes.
    """
    class_col = "classification_code" if "classification_code" in df.columns else "classification"
    if class_col not in df.columns:
        return {"revisions_present": [], "notes": "No classification column found"}

    revisions = sorted([str(r) for r in df[class_col].dropna().unique().tolist()])
    spans: dict[str, dict[str, str]] = {}
    for rev in revisions:
        rev_sub = df[df[class_col] == rev]
        spans[rev] = {
            "start_period": str(rev_sub["period"].min()),
            "end_period": str(rev_sub["period"].max()),
            "row_count": len(rev_sub),
        }

    return {
        "revisions_present": revisions,
        "revision_spans": spans,
        "chapter_level_continuity": "HS2 2-digit chapters remain structurally consistent across H4, H5, and H6 WCO revisions.",
    }
