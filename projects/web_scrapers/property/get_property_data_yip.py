from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import concurrent.futures
import pandas as pd
import IPython


def get_suburb_data(s, state):
    # Get HTML
    name = s.split("'>")[-1].split("</a>")[0]
    postcode = s.split(f"/{state}/")[-1].split("-")[0]
    s = s.split("' class='", 1)[0]
    s = s.split("href='", 1)[-1]
    link = f"https://www.yourinvestmentpropertymag.com.au{s}"
    r = requests.get(link)
    html = r.text  # the HTML code you've written above
    parsed_html = BeautifulSoup(html, features="html.parser")
    # Summary data
    summary_div = parsed_html.body.find("div", attrs={"id": "pills-house-report"})
    summary_text = summary_div.get_text().replace("\n", "").replace("  ", "")

    # Setting up data object
    data = {
        "suburb": name,
        "postcode": postcode,
        "summary": summary_text,
        "house_median_price": None,
        "unit_median_price": None,
        "house_quarterly_growth": None,
        "unit_quarterly_growth": None,
        "house_12m_growth": None,
        "unit_12m_growth": None,
        "house_avg_annual_growth": None,
        "unit_avg_annual_growth": None,
        "house_weekly_median_rent": None,
        "unit_weekly_median_rent": None,
        "house_gross_rental_yield": None,
        "unit_gross_rental_yield": None,
        "house_no_of_sales_12m": None,
        "unit_no_of_sales_12m": None,
        "house_avg_days_on_market_12m": None,
        "unit_avg_days_on_market_12m": None,
        "2011_total_population": None,
        "2016_total_population": None,
        "2011_population_change_5y_pc": None,
        "2016_population_change_5y_pc": None,
        "2011_median_household_income_pw": None,
        "2016_median_household_income_pw": None,
        "2011_household_income_change_5y_pc": None,
        "2016_household_income_change_5y_pc": None,
        "2011_median_age": None,
        "2016_median_age": None,
    }

    key_table_report_divs = parsed_html.body.find_all(
        "div", attrs={"class": "key-table-report"}
    )

    # Key market data report
    key_market_div = key_table_report_divs[0]
    tr = key_market_div.find_all("tr")
    for idx, r in enumerate(tr):
        td = r.find_all("td")

        if idx == 1:
            data["house_median_price"] = (
                str(td[1])
                .split('">')[-1]
                .split("</span>")[0]
                .split("</label>")[-1]
                .replace(",", "")
            )
            data["unit_median_price"] = (
                str(td[2])
                .split('">')[-1]
                .split("</span>")[0]
                .split("</label>")[-1]
                .replace(",", "")
            )
        elif idx == 2:
            data["house_quarterly_growth"] = str(td[1]).split('">')[-1].split("</")[0]
            data["unit_quarterly_growth"] = str(td[2]).split('">')[-1].split("</")[0]
        elif idx == 3:
            data["house_12m_growth"] = str(td[1]).split('">')[-1].split("</")[0]
            data["unit_12m_growth"] = str(td[2]).split('">')[-1].split("</")[0]
        elif idx == 4:
            data["house_avg_annual_growth"] = str(td[1]).split('">')[-1].split("</")[0]
            data["unit_avg_annual_growth"] = str(td[2]).split('">')[-1].split("</")[0]
        elif idx == 5:
            data["house_weekly_median_rent"] = (
                str(td[1])
                .split('">')[-1]
                .split("</span>")[0]
                .split("</label>")[-1]
                .replace(",", "")
            )
            data["unit_weekly_median_rent"] = (
                str(td[2])
                .split('">')[-1]
                .split("</span>")[0]
                .split("</label>")[-1]
                .replace(",", "")
            )
        elif idx == 6:
            data["house_gross_rental_yield"] = str(td[1]).split('">')[-1].split("</")[0]
            data["unit_gross_rental_yield"] = str(td[2]).split('">')[-1].split("</")[0]
        elif idx == 7:
            data["house_no_of_sales_12m"] = str(td[1]).split('">')[-1].split("</")[0]
            data["unit_no_of_sales_12m"] = str(td[2]).split('">')[-1].split("</")[0]
        elif idx == 8:
            data["house_avg_days_on_market_12m"] = (
                str(td[1]).split('">')[-1].split("</")[0]
            )
            data["unit_avg_days_on_market_12m"] = (
                str(td[2]).split('">')[-1].split("</")[0]
            )

    # Key demographics report
    key_demographics_div = key_table_report_divs[1]
    tr = key_demographics_div.find_all("tr")

    for idx, r in enumerate(tr):
        td = r.find_all("td")
        if idx == 1:
            data["2011_total_population"] = (
                str(td[1])
                .split("<span>")[-1]
                .split('no-value">')[-1]
                .split("</span>")[0]
                .replace(",", "")
            )
            data["2016_total_population"] = (
                str(td[2])
                .split("<span>")[-1]
                .split('no-value">')[-1]
                .split("</span>")[0]
                .replace(",", "")
            )
        elif idx == 2:
            data["2011_population_change_5y_pc"] = (
                str(td[1])
                .split("<span>")[-1]
                .split('no-value">')[-1]
                .split("</span>")[0]
            )
            data["2016_population_change_5y_pc"] = (
                str(td[2])
                .split("<span>")[-1]
                .split('no-value">')[-1]
                .split("</span>")[0]
            )
        elif idx == 3:
            data["2011_median_household_income_pw"] = (
                str(td[1])
                .split('">')[-1]
                .split("</span>")[0]
                .split("<span>")[-1]
                .split("</label>")[-1]
                .replace(",", "")
                .replace("$", "")
            )
            data["2016_median_household_income_pw"] = (
                str(td[2])
                .split('">')[-1]
                .split("</span>")[0]
                .split("<span>")[-1]
                .split("</label>")[-1]
                .replace(",", "")
                .replace("$", "")
            )
        elif idx == 4:
            data["2011_household_income_change_5y_pc"] = (
                str(td[1])
                .split("<span>")[-1]
                .split('no-value">')[-1]
                .split("</span>")[0]
            )
            data["2016_household_income_change_5y_pc"] = (
                str(td[2])
                .split("<span>")[-1]
                .split('no-value">')[-1]
                .split("</span>")[0]
            )
        elif idx == 5:
            data["2011_median_age"] = (
                str(td[1])
                .split("<span>")[-1]
                .split('no-value">')[-1]
                .split("</span>")[0]
            )
            data["2016_median_age"] = (
                str(td[2])
                .split("<span>")[-1]
                .split('no-value">')[-1]
                .split("</span>")[0]
            )

    total_data.append(data)


if __name__ == "__main__":
    total_data, suburb_list = [], []
    num_workers = 20
    state = "act"
    data_file = f"metadata/{state}_suburbs.txt"

    # Create list from txt file where every entry is a new object
    with open(data_file, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            suburb_list.append(line)
    state_list = [state for i in range(len(suburb_list))]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        _ = tqdm(
            executor.map(get_suburb_data, suburb_list, state_list),
            total=len(suburb_list),
        )
    df = pd.DataFrame(total_data)

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # more options can be specified also
        print(df)

    df.to_csv(f"property_data_{state}_2022_2016_census.tsv", sep="\t")
