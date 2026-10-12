"""Step 1: download raw data from WRDS (requires a WRDS account; pip install wrds).
Output: parquet files in data/raw/."""
import pandas as pd
from .config import RAW, DATA_START

FUNDQ_STOCK = ["atq", "ltq", "ceqq", "seqq", "pstkq", "txditcq", "cheq", "actq",
               "lctq", "ppentq", "invtq", "rectq", "apq", "dlttq", "dlcq",
               "icaptq", "mibq"]
FUNDQ_FLOW = ["saleq", "cogsq", "xsgaq", "xrdq", "dpq", "xintq", "oiadpq",
              "oibdpq", "piq", "txtq", "ibq", "ibcomq", "ibadjq", "nopiq",
              "spiq", "miiq"]
FUNDQ_YTD = ["capxy", "oancfy", "dvy"]   # year-to-date items, converted to single quarters later


def download_all(username: str, end: str = "2020-12-31"):
    import wrds
    RAW.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection(wrds_username=username)

    msf = db.raw_sql(f"""
        select permno, date, prc, shrout, ret, retx
        from crsp.msf where date between '{DATA_START}' and '{end}'""",
        date_cols=["date"])
    msf.to_parquet(RAW / "crsp_msf.parquet")

    names = db.raw_sql("""
        select permno, namedt, nameendt, shrcd, exchcd, siccd, ticker, comnam
        from crsp.msenames""", date_cols=["namedt", "nameendt"])
    names.to_parquet(RAW / "crsp_msenames.parquet")

    delist = db.raw_sql("select permno, dlstdt, dlret from crsp.msedelist",
                        date_cols=["dlstdt"])
    delist.to_parquet(RAW / "crsp_msedelist.parquet")

    link = db.raw_sql("""
        select gvkey, lpermno as permno, linktype, linkprim, linkdt, linkenddt
        from crsp.ccmxpf_linktable
        where linktype in ('LU','LC') and linkprim in ('P','C')""",
        date_cols=["linkdt", "linkenddt"])
    link.to_parquet(RAW / "ccm_link.parquet")

    cols = ", ".join(FUNDQ_STOCK + FUNDQ_FLOW + FUNDQ_YTD)
    fundq = db.raw_sql(f"""
        select gvkey, datadate, rdq, fyearq, fqtr, {cols}
        from comp.fundq
        where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
          and datadate between '{DATA_START}' and '{end}'""",
        date_cols=["datadate", "rdq"])
    fundq.to_parquet(RAW / "comp_fundq.parquet")

    # gvkey <-> CIK map: matches WRDS and free-data results firm by firm (rv/compare.py)
    comp = db.raw_sql("select gvkey, cik, conm from comp.company")
    comp.to_parquet(RAW / "comp_company.parquet")
    db.close()
    print("Done. Also download the Fama-French factor CSVs and Siccodes49.txt (see README).")
