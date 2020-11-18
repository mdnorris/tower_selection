#
import timeit
import datetime
import numpy as np
import pandas as pd
import cudf, io, requests
from io import StringIO

url = "https://github.com/plotly/datasets/raw/master/tips.csv"
content = requests.get(url).content.decode('utf-8')

tips_df = cudf.read_csv(StringIO(content))
tips_df['tip_percentage'] = tips_df['tip'] / tips_df['total_bill'] * 100

# display average tip by dining party size
print(tips_df.groupby('size').tip_percentage.mean())
init_time = timeit.default_timer()
init_time_2 = timeit.default_timer()
fasdafsd
# SECTION 1: DATA INPUTS
sites = pd.read_csv("site_daily_small.csv", delimiter=",")
hr_sites = pd.read_csv("bin_hourly.csv", delimiter=",")
npv_values = pd.read_csv("npv.csv", delimiter=",")
lte_params = pd.read_csv('lte_assumptions.csv', delimiter=',')

# sites = sites.sample(frac=.3, random_state=1)
# hr_sites = hr_sites.sample(frac=.3, random_state=1)

# candidate dropping criteria
npv_threshold = 0
bin_prox = 17
sinr_deg = 10

sites = sites.dropna()
hr_sites = hr_sites.dropna()
first_sites = sites

rb_bins = pd.merge(sites, hr_sites, on='GridName')
rb_bins['bin_req_hr_mbps'] = (rb_bins['Hour_GBs'] * 8 * (2 ** 10)) / 3600
sum_req = rb_bins.sort_values(by=['GridName', 'bin_req_hr_mbps'], ascending=[True, False])
bin_req = rb_bins.groupby('GridName')['bin_req_hr_mbps'].max().reset_index()

day_usage = hr_sites.groupby("GridName")["Hour_GBs"].sum().reset_index()
sites = pd.merge(sites, day_usage, on="GridName")

# NPV DATA
npv_values.columns = [
    "non",
    "pole_du",
    "pole_u",
    "pole_s",
    "",
    "off_du",
    "off_u",
    "off_s",
    "off_r",
    "roe_du",
    "roe_u",
    "roe_s",
    "roe_r",
    "smb_du",
    "smb_u",
    "smb_s",
    "smb_r",
]
disc_rt = 0.15

# LTE INPUTS
rx_sensitivity_db = -89.4


# SECTION 2: FUNCTIONS


def asset(asset_in):
    asset_type = 0
    if asset_in == "Pole":
        asset_type = 1
    return asset_type


def morph(morph_in):
    morph_type = 0
    if morph_in == "Dense Urban":
        morph_type = 11
    elif morph_in == "Urban":
        morph_type = 13
    elif morph_in == "Suburban":
        morph_type = 17
    return morph_type


morph_pairs = {
    11: 1,
    13: 2,
    17: 3,
    19: 4,
    55: 9,
    65: 10,
    85: 11,
    95: 12,
    33: 13,
    39: 14,
    51: 15,
    57: 16,
    77: 5,
    91: 6,
    119: 7,
    133: 8,
}


def asset_morph(asset_num, morph_num):
    a_m = asset_num * morph_num
    return morph_pairs[a_m]


def morph_array_pole(code):
    ma_pole = ''
    if code == 1:
        ma_pole = pole_du
    elif code == 2:
        ma_pole = pole_u
    elif code == 3:
        ma_pole = pole_s
    return ma_pole


def fin_arrays_pole(code):
    cpx_pole = np.empty([11, 1])
    opx_pole = np.empty([11, 1])
    growth_pole = np.empty([11, 1])
    mvno_pole = np.empty([11, 1])
    npv_array_pole = np.zeros([len(npv_values), 1])
    if code == 1:
        npv_array_pole = npv_values.loc[:, "pole_du"]
    elif code == 2:
        npv_array_pole = npv_values.loc[:, "pole_u"]
    elif code == 3:
        npv_array_pole = npv_values.loc[:, "pole_s"]
    elif code == 4:
        npv_array_pole = npv_values.loc[:, ""]
    for m in range(11):
        cpx_pole[m] = float(npv_array_pole[m + 1])
    for m in range(11):
        opx_pole[m] = float(npv_array_pole[m + 12])
    for m in range(11):
        growth_pole[m] = float(npv_array_pole[m + 23])
    for m in range(11):
        mvno_pole[m] = float(npv_array_pole[m + 34])
    end_array = np.hstack((cpx_pole, opx_pole, growth_pole, mvno_pole))
    return end_array


def bld_npv21(gbs, code):
    array = morph_array_pole(code)
    array = array[1:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (
            1.15 ** 1
    )
    if value > 0:
        year = 1
    elif value < 0:
        year = 12
    else:
        year = 11
    return year


def bld_npv22(gbs, code):
    array = morph_array_pole(code)
    array = array[2:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (
            1.15 ** 2
    )
    if value > 0:
        year = 1
    elif value < 0:
        year = 12
    else:
        year = 11
    return year


def bld_npv23(gbs, code):
    array = morph_array_pole(code)
    array = array[3:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (
            1.15 ** 3
    )
    if value > 0:
        year = 1
    elif value < 0:
        year = 12
    else:
        year = 11
    return year


def bld_npv24(gbs, code):
    array = morph_array_pole(code)
    array = array[4:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (
            1.15 ** 4
    )
    if value > 0:
        year = 1
    elif value < 0:
        year = 12
    else:
        year = 11
    return year


def bld_npv25(gbs, code):
    array = morph_array_pole(code)
    array = array[5:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (
            1.15 ** 5
    )
    if value > 0:
        year = 1
    elif value < 0:
        year = 12
    else:
        year = 11
    return year


def bld_npv26(gbs, code):
    array = morph_array_pole(code)
    array = array[6:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (
            1.15 ** 6
    )
    if value > 0:
        year = 1
    elif value < 0:
        year = 12
    else:
        year = 11
    return year


def bld_npv27(gbs, code):
    array = morph_array_pole(code)
    array = array[7:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (
            1.15 ** 7
    )
    if value > 0:
        year = 1
    elif value < 0:
        year = 12
    else:
        year = 11
    return year


def bld_npv28(gbs, code):
    array = morph_array_pole(code)
    array = array[8:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (
            1.15 ** 8
    )
    if value > 0:
        year = 1
    elif value < 0:
        year = 12
    else:
        year = 11
    return year


def bld_npv29(gbs, code):
    array = morph_array_pole(code)
    array = array[9:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (
            1.15 ** 9
    )
    if value > 0:
        year = 1
    elif value < 0:
        year = 12
    else:
        year = 11
    return year


def bld_npv30(gbs, code):
    array = morph_array_pole(code)
    array = array[10:11, :]
    value = (gbs * 365 * array[0][3] * array[0][2] - array[0][1] - array[0][0]) / (
            1.15 ** 10
    )
    if value > 0:
        year = 1
    elif value < 0:
        year = 12
    else:
        year = 11
    return year


def loop_npv21(gbs, code):
    value = np.empty([10, 1])
    year = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    array = morph_array_pole(code)
    for m in range(10):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(
        sum(
            [
                value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                for value_i, year_i in zip(value, year)
            ]
        )
    )


def loop_npv22(gbs, code):
    value = np.empty([9, 1])
    year = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    array = morph_array_pole(code)
    for m in range(9):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(
        sum(
            [
                value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                for value_i, year_i in zip(value, year)
            ]
        )
    )


def loop_npv23(gbs, code):
    value = np.empty([8, 1])
    year = np.array([3, 4, 5, 6, 7, 8, 9, 10])
    array = morph_array_pole(code)
    for m in range(8):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(
        sum(
            [
                value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                for value_i, year_i in zip(value, year)
            ]
        )
    )


def loop_npv24(gbs, code):
    value = np.empty([7, 1])
    year = np.array([4, 5, 6, 7, 8, 9, 10])
    array = morph_array_pole(code)
    for m in range(7):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(
        sum(
            [
                value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                for value_i, year_i in zip(value, year)
            ]
        )
    )


def loop_npv25(gbs, code):
    value = np.empty([6, 1])
    year = np.array([5, 6, 7, 8, 9, 10])
    array = morph_array_pole(code)
    for m in range(6):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    npv_loop = int(
        sum(
            [
                value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                for value_i, year_i in zip(value, year)
            ]
        )
    )
    return npv_loop


def loop_npv26(gbs, code):
    value = np.empty([5, 1])
    year = np.array([6, 7, 8, 9, 10])
    array = morph_array_pole(code)
    for m in range(5):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    npv_loop = int(
        sum(
            [
                value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                for value_i, year_i in zip(value, year)
            ]
        )
    )
    return npv_loop


def loop_npv27(gbs, code):
    value = np.empty([4, 1])
    year = np.array([7, 8, 9, 10])
    array = morph_array_pole(code)
    for m in range(4):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(
        sum(
            [
                value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                for value_i, year_i in zip(value, year)
            ]
        )
    )


def loop_npv28(gbs, code):
    value = np.empty([3, 1])
    year = np.array([8, 9, 10])
    array = morph_array_pole(code)
    for m in range(3):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(
        sum(
            [
                value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                for value_i, year_i in zip(value, year)
            ]
        )
    )


def loop_npv29(gbs, code):
    value = np.empty([2, 1])
    year = np.array([9, 10])
    array = morph_array_pole(code)
    for m in range(2):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(
        sum(
            [
                value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                for value_i, year_i in zip(value, year)
            ]
        )
    )


def loop_npv30(gbs, code):
    value = np.empty([1, 1])
    year = np.array([10])
    array = morph_array_pole(code)
    for m in range(1):
        value[m] = gbs * 365 * array[m][3] * array[m][2] - array[m][1]
    year_0 = 0
    value[0] = value[0] - array[0][0]
    return int(
        sum(
            [
                value_i / ((1.0 + disc_rt) ** (year_i - year_0))
                for value_i, year_i in zip(value, year)
            ]
        )
    )


def rx_calc(path_loss_umi_db):
    rx_signal_strength_db = 37.0 - path_loss_umi_db
    sinr = (rx_signal_strength_db - rx_sensitivity_db).round(0).astype(int)
    rx_signal_strength_mw = 10 ** ((37.0 - path_loss_umi_db) / 10) / 1000.000000000
    return rx_signal_strength_db, sinr, rx_signal_strength_mw


def sinr_new(sum_rx_signal, rx_signal_strength_db):
    # note that -89.4 here is rx_sensitivity_db
    sinr_nouveau = np.round(
        (
            sum_rx_signal.where(
                sum_rx_signal.isnull(),
                rx_signal_strength_db
                - (np.log10((sum_rx_signal + (10 ** (-89.4 / 10)) / 1000) * 1000) * 10),
            )
        ),
        0,
    )
    return sinr_nouveau


# SECTION 3: FIRST ITERATION TO GENERATE INHERENT NPV

pole_du = fin_arrays_pole(1)
pole_u = fin_arrays_pole(2)
pole_s = fin_arrays_pole(3)

sites["asset_id"] = sites.apply(lambda x: asset(x["Type"]), axis=1)
sites["morph_id"] = sites.apply(lambda x: morph(x["Morphology"]), axis=1)
sites["morph_code"] = sites.apply(
    lambda x: asset_morph(x["asset_id"], x["morph_id"]), axis=1
)
sites = sites.drop(["Type", "Morphology", "asset_id", "morph_id"], axis=1)
site_t_m = sites.groupby("fict_site")["morph_code"].mean().reset_index()
init_sites = sites

# calculation of build years from inherent npv
site_bin_gbs = sites.groupby("fict_site")["Hour_GBs"].sum().reset_index()
bld_yr_sites = pd.merge(site_bin_gbs, site_t_m, on="fict_site")

bld_yr_sites["1"] = bld_yr_sites.apply(
    lambda x: bld_npv21(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["2"] = bld_yr_sites.apply(
    lambda x: bld_npv22(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["3"] = bld_yr_sites.apply(
    lambda x: bld_npv23(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["4"] = bld_yr_sites.apply(
    lambda x: bld_npv24(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["5"] = bld_yr_sites.apply(
    lambda x: bld_npv25(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["6"] = bld_yr_sites.apply(
    lambda x: bld_npv26(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["7"] = bld_yr_sites.apply(
    lambda x: bld_npv27(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["8"] = bld_yr_sites.apply(
    lambda x: bld_npv28(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["9"] = bld_yr_sites.apply(
    lambda x: bld_npv29(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["10"] = bld_yr_sites.apply(
    lambda x: bld_npv30(x["Hour_GBs"], x["morph_code"]), axis=1
)

inh_bld_yr = bld_yr_sites[["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]]

# chooses min and excludes any negative npv values
bld_yr_sites["build_yr"] = inh_bld_yr.idxmin(axis=1).astype(int)
bld_yr_sites = bld_yr_sites[bld_yr_sites["10"] != 12]
print(bld_yr_sites['build_yr'].value_counts())
init_build_yr_sites = bld_yr_sites[["fict_site", "build_yr"]]

sites_yr1 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 1]
sites_yr2 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 2]
sites_yr3 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 3]
sites_yr4 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 4]
sites_yr5 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 5]
sites_yr6 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 6]
sites_yr7 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 7]
sites_yr8 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 8]
sites_yr9 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 9]
sites_yr10 = bld_yr_sites.loc[bld_yr_sites["build_yr"] == 10]

site_number = (
        len(sites_yr1)
        + len(sites_yr2)
        + len(sites_yr3)
        + len(sites_yr4)
        + len(sites_yr5)
        + len(sites_yr6)
        + len(sites_yr7)
        + len(sites_yr8)
        + len(sites_yr9)
        + len(sites_yr10)
)

selected = pd.DataFrame([])

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]

(
    init_sites["rx_signal_strength_db"],
    init_sites["sinr"],
    init_sites["rx_signal_strength_mw"],
) = rx_calc(init_sites["path_loss_umi_db"])
init_sites.loc[init_sites["sinr"] > 50, "sinr"] = 50.0
calc_sites = init_sites[
    [
        "fict_site",
        "GridName",
        "Hour_GBs",
        "sinr",
        "rx_signal_strength_db",
        "rx_signal_strength_mw",
        "morph_code",
    ]
]

# note that declaring variables and following if statement avoid errors
sites_yr1 = sites_yr1[["fict_site"]]
sites_1 = pd.merge(calc_sites, sites_yr1, on="fict_site")
start_yr1 = timeit.default_timer()
candidates = pd.DataFrame([])
init_ranking = pd.DataFrame([])
bad_candidate = 0.0
full = 1

if len(sites_yr1) > 0:
    sites = sites_1
    sites = sites.groupby("fict_site", as_index=False).agg(
        {"morph_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv21(x["Hour_GBs"], x["morph_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fict_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_1, on="fict_site")
    init_ranking_counts = init_ranking_bins["GridName"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "GridName", "GridName": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="GridName")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(selected, sort=True)
        selected = selected[["fict_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fict_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on="fict_site")
        sinr_bins_unique = sinr_bins[sinr_bins["bin_count"] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins["bin_count"] > 1].copy()
        temp_sinr_bins_dups = (
            sinr_bins_dups.groupby("GridName")["rx_signal_strength_mw"]
                .sum()
                .reset_index()
        )
        sinr_bins_dups = sinr_bins_dups.drop(["rx_signal_strength_mw"], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on="GridName")
        sinr_bins_unique["sinr_new"] = sinr_bins_unique["sinr"]
        sinr_bins_dups["sinr_new"] = sinr_new(
            sinr_bins_dups["rx_signal_strength_mw"],
            sinr_bins_dups["rx_signal_strength_db"],
        )
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby("fict_site", as_index=False).agg(
            {"morph_code": "mean", "Hour_GBs": "sum"}
        )
        sinr_bins_agg["xnpv"] = sinr_bins_agg.apply(
            lambda x: loop_npv21(x["Hour_GBs"], x["morph_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = sinr_bins_agg["xnpv"].min()
        next_best["sum_npv"] = sinr_bins_agg["xnpv"].sum()
        next_best = next_best[["fict_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= npv_threshold:
            bad_candidate = next_best["fict_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fict_site")
        init_cand_bins = init_cand_bins.nlargest(bin_prox, "sinr")
        init_cand_bins = init_cand_bins[["sinr", "fict_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_sinr_bins, on="fict_site")
        cand_bins = cand_bins.nlargest(bin_prox, "sinr_new")
        cand_bins = cand_bins[["sinr_new"]].mean()
        if (abs(init_cand_bins["sinr"] - cand_bins["sinr_new"])) > sinr_deg:
            bad_candidate = init_cand_bins["fict_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fict_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fict_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fict_site"})

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]

start_yr2 = timeit.default_timer()
sites_yr2 = sites_yr2[["fict_site"]]
sites_2 = pd.merge(calc_sites, sites_yr2, on="fict_site")

if len(sites_yr2) > 0:
    sites = sites_2
    sites = sites.groupby("fict_site", as_index=False).agg(
        {"morph_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv22(x["Hour_GBs"], x["morph_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fict_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_2, on="fict_site")
    init_ranking_counts = init_ranking_bins["GridName"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "GridName", "GridName": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="GridName")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(selected, sort=True)
        selected = selected[["fict_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fict_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on="fict_site")
        sinr_bins_unique = sinr_bins[sinr_bins["bin_count"] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins["bin_count"] > 1].copy()
        temp_sinr_bins_dups = (
            sinr_bins_dups.groupby("GridName")["rx_signal_strength_mw"]
                .sum()
                .reset_index()
        )
        sinr_bins_dups = sinr_bins_dups.drop(["rx_signal_strength_mw"], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on="GridName")
        sinr_bins_unique["sinr_new"] = sinr_bins_unique["sinr"]
        sinr_bins_dups["sinr_new"] = sinr_new(
            sinr_bins_dups["rx_signal_strength_mw"],
            sinr_bins_dups["rx_signal_strength_db"],
        )
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby("fict_site", as_index=False).agg(
            {"morph_code": "mean", "Hour_GBs": "sum"}
        )
        sinr_bins_agg["xnpv"] = sinr_bins_agg.apply(
            lambda x: loop_npv22(x["Hour_GBs"], x["morph_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = sinr_bins_agg["xnpv"].min()
        next_best["sum_npv"] = sinr_bins_agg["xnpv"].sum()
        next_best = next_best[["fict_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= npv_threshold:
            bad_candidate = next_best["fict_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fict_site")
        init_cand_bins = init_cand_bins.nlargest(bin_prox, "sinr")
        init_cand_bins = init_cand_bins[["sinr", "fict_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_sinr_bins, on="fict_site")
        cand_bins = cand_bins.nlargest(bin_prox, "sinr_new")
        cand_bins = cand_bins[["sinr_new"]].mean()
        if (abs(init_cand_bins["sinr"] - cand_bins["sinr_new"])) > sinr_deg:
            bad_candidate = init_cand_bins["fict_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fict_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fict_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fict_site"})
    elif len(candidates) == 0:
        break

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]

sites_yr3 = sites_yr3[["fict_site"]]
sites_3 = pd.merge(calc_sites, sites_yr3, on="fict_site")

if len(sites_yr3) > 0:
    sites = sites_3
    sites = sites.groupby("fict_site", as_index=False).agg(
        {"morph_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv23(x["Hour_GBs"], x["morph_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fict_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_3, on="fict_site")
    init_ranking_counts = init_ranking_bins["GridName"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "GridName", "GridName": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="GridName")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fict_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fict_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on="fict_site")

        sinr_bins_unique = sinr_bins[sinr_bins["bin_count"] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins["bin_count"] > 1].copy()
        temp_sinr_bins_dups = (
            sinr_bins_dups.groupby("GridName")["rx_signal_strength_mw"]
                .sum()
                .reset_index()
        )
        sinr_bins_dups = sinr_bins_dups.drop(["rx_signal_strength_mw"], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on="GridName")
        sinr_bins_unique["sinr_new"] = sinr_bins_unique["sinr"]
        sinr_bins_dups["sinr_new"] = sinr_new(
            sinr_bins_dups["rx_signal_strength_mw"],
            sinr_bins_dups["rx_signal_strength_db"],
        )
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby("fict_site", as_index=False).agg(
            {"morph_code": "mean", "Hour_GBs": "sum"}
        )
        sinr_bins_agg["xnpv"] = sinr_bins_agg.apply(
            lambda x: loop_npv23(x["Hour_GBs"], x["morph_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = sinr_bins_agg["xnpv"].min()
        next_best["sum_npv"] = sinr_bins_agg["xnpv"].sum()
        next_best = next_best[["fict_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= npv_threshold:
            bad_candidate = next_best["fict_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fict_site")
        init_cand_bins = init_cand_bins.nlargest(bin_prox, "sinr")
        init_cand_bins = init_cand_bins[["sinr", "fict_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_sinr_bins, on="fict_site")
        cand_bins = cand_bins.nlargest(bin_prox, "sinr_new")
        cand_bins = cand_bins[["sinr_new"]].mean()
        if (abs(init_cand_bins["sinr"] - cand_bins["sinr_new"])) > sinr_deg:
            bad_candidate = init_cand_bins["fict_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fict_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fict_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fict_site"})
    elif len(candidates) == 0:
        break

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]

sites_yr4 = sites_yr4[["fict_site"]]
sites_4 = pd.merge(calc_sites, sites_yr4, on="fict_site")

if len(sites_yr4) > 0:
    sites = sites_4
    sites = sites.groupby("fict_site", as_index=False).agg(
        {"morph_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv24(x["Hour_GBs"], x["morph_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fict_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_4, on="fict_site")
    init_ranking_counts = init_ranking_bins["GridName"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "GridName", "GridName": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="GridName")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fict_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fict_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on="fict_site")
        sinr_bins_unique = sinr_bins[sinr_bins["bin_count"] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins["bin_count"] > 1].copy()
        temp_sinr_bins_dups = (
            sinr_bins_dups.groupby("GridName")["rx_signal_strength_mw"]
                .sum()
                .reset_index()
        )
        sinr_bins_dups = sinr_bins_dups.drop(["rx_signal_strength_mw"], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on="GridName")
        sinr_bins_unique["sinr_new"] = sinr_bins_unique["sinr"]
        sinr_bins_dups["sinr_new"] = sinr_new(
            sinr_bins_dups["rx_signal_strength_mw"],
            sinr_bins_dups["rx_signal_strength_db"],
        )
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]
        sinr_bins_agg = sinr_bins.groupby("fict_site", as_index=False).agg(
            {"morph_code": "mean", "Hour_GBs": "sum"}
        )
        sinr_bins_agg["xnpv"] = sinr_bins_agg.apply(
            lambda x: loop_npv24(x["Hour_GBs"], x["morph_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = sinr_bins_agg["xnpv"].min()
        next_best["sum_npv"] = sinr_bins_agg["xnpv"].sum()
        next_best = next_best[["fict_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= npv_threshold:
            bad_candidate = next_best["fict_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fict_site")
        init_cand_bins = init_cand_bins.nlargest(bin_prox, "sinr")
        init_cand_bins = init_cand_bins[["sinr", "fict_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_sinr_bins, on="fict_site")
        cand_bins = cand_bins.nlargest(bin_prox, "sinr_new")
        cand_bins = cand_bins[["sinr_new"]].mean()
        if (abs(init_cand_bins["sinr"] - cand_bins["sinr_new"])) > sinr_deg:
            bad_candidate = init_cand_bins["fict_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fict_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fict_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fict_site"})

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]

sites_yr5 = sites_yr5[["fict_site"]]
sites_5 = pd.merge(calc_sites, sites_yr5, on="fict_site")

if len(sites_yr5) > 0:
    sites = sites_5
    sites = sites.groupby("fict_site", as_index=False).agg(
        {"morph_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv25(x["Hour_GBs"], x["morph_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fict_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_5, on="fict_site")
    init_ranking_counts = init_ranking_bins["GridName"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "GridName", "GridName": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="GridName")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fict_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fict_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on="fict_site")
        sinr_bins_unique = sinr_bins[sinr_bins["bin_count"] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins["bin_count"] > 1].copy()
        temp_sinr_bins_dups = (
            sinr_bins_dups.groupby("GridName")["rx_signal_strength_mw"]
                .sum()
                .reset_index()
        )
        sinr_bins_dups = sinr_bins_dups.drop(["rx_signal_strength_mw"], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on="GridName")
        sinr_bins_unique["sinr_new"] = sinr_bins_unique["sinr"]
        sinr_bins_dups["sinr_new"] = sinr_new(
            sinr_bins_dups["rx_signal_strength_mw"],
            sinr_bins_dups["rx_signal_strength_db"],
        )
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby("fict_site", as_index=False).agg(
            {"morph_code": "mean", "Hour_GBs": "sum"}
        )
        sinr_bins_agg["xnpv"] = sinr_bins_agg.apply(
            lambda x: loop_npv25(x["Hour_GBs"], x["morph_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = sinr_bins_agg["xnpv"].min()
        next_best["sum_npv"] = sinr_bins_agg["xnpv"].sum()
        next_best = next_best[["fict_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= npv_threshold:
            bad_candidate = next_best["fict_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fict_site")
        init_cand_bins = init_cand_bins.nlargest(bin_prox, "sinr")
        init_cand_bins = init_cand_bins[["sinr", "fict_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_sinr_bins, on="fict_site")
        cand_bins = cand_bins.nlargest(bin_prox, "sinr_new")
        cand_bins = cand_bins[["sinr_new"]].mean()
        if (abs(init_cand_bins["sinr"] - cand_bins["sinr_new"])) > sinr_deg:
            bad_candidate = init_cand_bins["fict_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fict_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fict_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fict_site"})

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]

sites_yr6 = sites_yr6[["fict_site"]]
sites_6 = pd.merge(calc_sites, sites_yr6, on="fict_site")

if len(sites_yr6) > 0:
    sites = sites_6
    sites = sites.groupby("fict_site", as_index=False).agg(
        {"morph_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv26(x["Hour_GBs"], x["morph_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fict_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_6, on="fict_site")
    init_ranking_counts = init_ranking_bins["GridName"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "GridName", "GridName": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="GridName")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fict_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fict_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on="fict_site")
        sinr_bins_unique = sinr_bins[sinr_bins["bin_count"] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins["bin_count"] > 1].copy()
        temp_sinr_bins_dups = (
            sinr_bins_dups.groupby("GridName")["rx_signal_strength_mw"]
                .sum()
                .reset_index()
        )
        sinr_bins_dups = sinr_bins_dups.drop(["rx_signal_strength_mw"], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on="GridName")
        sinr_bins_unique["sinr_new"] = sinr_bins_unique["sinr"]
        sinr_bins_dups["sinr_new"] = sinr_new(
            sinr_bins_dups["rx_signal_strength_mw"],
            sinr_bins_dups["rx_signal_strength_db"],
        )
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby("fict_site", as_index=False).agg(
            {"morph_code": "mean", "Hour_GBs": "sum"}
        )
        sinr_bins_agg["xnpv"] = sinr_bins_agg.apply(
            lambda x: loop_npv26(x["Hour_GBs"], x["morph_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = sinr_bins_agg["xnpv"].min()
        next_best["sum_npv"] = sinr_bins_agg["xnpv"].sum()
        next_best = next_best[["fict_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= npv_threshold:
            bad_candidate = next_best["fict_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fict_site")
        init_cand_bins = init_cand_bins.nlargest(bin_prox, "sinr")
        init_cand_bins = init_cand_bins[["sinr", "fict_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_sinr_bins, on="fict_site")
        cand_bins = cand_bins.nlargest(bin_prox, "sinr_new")
        cand_bins = cand_bins[["sinr_new"]].mean()
        if (abs(init_cand_bins["sinr"] - cand_bins["sinr_new"])) > sinr_deg:
            bad_candidate = init_cand_bins["fict_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fict_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fict_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fict_site"})

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]

start_yr7 = timeit.default_timer()
sites_yr7 = sites_yr7[["fict_site"]]
sites_7 = pd.merge(calc_sites, sites_yr7, on="fict_site")

if len(sites_yr7) > 0:
    sites = sites_7
    sites = sites.groupby("fict_site", as_index=False).agg(
        {"morph_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv27(x["Hour_GBs"], x["morph_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fict_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_7, on="fict_site")
    init_ranking_counts = init_ranking_bins["GridName"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "GridName", "GridName": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="GridName")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fict_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fict_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on="fict_site")
        sinr_bins_unique = sinr_bins[sinr_bins["bin_count"] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins["bin_count"] > 1].copy()
        temp_sinr_bins_dups = (
            sinr_bins_dups.groupby("GridName")["rx_signal_strength_mw"]
                .sum()
                .reset_index()
        )
        sinr_bins_dups = sinr_bins_dups.drop(["rx_signal_strength_mw"], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on="GridName")
        sinr_bins_unique["sinr_new"] = sinr_bins_unique["sinr"]
        sinr_bins_dups["sinr_new"] = sinr_new(
            sinr_bins_dups["rx_signal_strength_mw"],
            sinr_bins_dups["rx_signal_strength_db"],
        )
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby("fict_site", as_index=False).agg(
            {"morph_code": "mean", "Hour_GBs": "sum"}
        )
        sinr_bins_agg["xnpv"] = sinr_bins_agg.apply(
            lambda x: loop_npv27(x["Hour_GBs"], x["morph_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = sinr_bins_agg["xnpv"].min()
        next_best["sum_npv"] = sinr_bins_agg["xnpv"].sum()
        next_best = next_best[["fict_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= npv_threshold:
            bad_candidate = next_best["fict_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fict_site")
        init_cand_bins = init_cand_bins.nlargest(bin_prox, "sinr")
        init_cand_bins = init_cand_bins[["sinr", "fict_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_sinr_bins, on="fict_site")
        cand_bins = cand_bins.nlargest(bin_prox, "sinr_new")
        cand_bins = cand_bins[["sinr_new"]].mean()
        if (abs(init_cand_bins["sinr"] - cand_bins["sinr_new"])) > sinr_deg:
            bad_candidate = init_cand_bins["fict_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fict_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fict_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fict_site"})

end_yr7 = timeit.default_timer()

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]

start_yr8 = timeit.default_timer()
sites_yr8 = sites_yr8[["fict_site"]]
sites_8 = pd.merge(calc_sites, sites_yr8, on="fict_site")

if len(sites_yr8) > 0:
    sites = sites_8
    sites = sites.groupby("fict_site", as_index=False).agg(
        {"morph_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv28(x["Hour_GBs"], x["morph_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fict_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_8, on="fict_site")
    init_ranking_counts = init_ranking_bins["GridName"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "GridName", "GridName": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="GridName")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fict_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fict_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on="fict_site")
        sinr_bins_unique = sinr_bins[sinr_bins["bin_count"] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins["bin_count"] > 1].copy()
        temp_sinr_bins_dups = (
            sinr_bins_dups.groupby("GridName")["rx_signal_strength_mw"]
                .sum()
                .reset_index()
        )
        sinr_bins_dups = sinr_bins_dups.drop(["rx_signal_strength_mw"], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on="GridName")
        sinr_bins_unique["sinr_new"] = sinr_bins_unique["sinr"]
        sinr_bins_dups["sinr_new"] = sinr_new(
            sinr_bins_dups["rx_signal_strength_mw"],
            sinr_bins_dups["rx_signal_strength_db"],
        )
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby("fict_site", as_index=False).agg(
            {"morph_code": "mean", "Hour_GBs": "sum"}
        )
        sinr_bins_agg["xnpv"] = sinr_bins_agg.apply(
            lambda x: loop_npv28(x["Hour_GBs"], x["morph_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = sinr_bins_agg["xnpv"].min()
        next_best["sum_npv"] = sinr_bins_agg["xnpv"].sum()
        next_best = next_best[["fict_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= npv_threshold:
            bad_candidate = next_best["fict_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fict_site")
        init_cand_bins = init_cand_bins.nlargest(bin_prox, "sinr")
        init_cand_bins = init_cand_bins[["sinr", "fict_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_sinr_bins, on="fict_site")
        cand_bins = cand_bins.nlargest(bin_prox, "sinr_new")
        cand_bins = cand_bins[["sinr_new"]].mean()
        if (abs(init_cand_bins["sinr"] - cand_bins["sinr_new"])) > sinr_deg:
            bad_candidate = init_cand_bins["fict_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fict_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fict_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fict_site"})

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]

sites_yr9 = sites_yr9[["fict_site"]]
sites_9 = pd.merge(calc_sites, sites_yr9, on="fict_site")

if len(sites_yr9) > 0:
    sites = sites_9
    sites = sites.groupby("fict_site", as_index=False).agg(
        {"morph_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv29(x["Hour_GBs"], x["morph_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fict_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_9, on="fict_site")
    init_ranking_counts = init_ranking_bins["GridName"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "GridName", "GridName": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="GridName")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fict_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fict_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on="fict_site")
        sinr_bins_unique = sinr_bins[sinr_bins["bin_count"] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins["bin_count"] > 1].copy()
        temp_sinr_bins_dups = (
            sinr_bins_dups.groupby("GridName")["rx_signal_strength_mw"]
                .sum()
                .reset_index()
        )
        sinr_bins_dups = sinr_bins_dups.drop(["rx_signal_strength_mw"], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on="GridName")
        sinr_bins_unique["sinr_new"] = sinr_bins_unique["sinr"]
        sinr_bins_dups["sinr_new"] = sinr_new(
            sinr_bins_dups["rx_signal_strength_mw"],
            sinr_bins_dups["rx_signal_strength_db"],
        )
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]

        sinr_bins_agg = sinr_bins.groupby("fict_site", as_index=False).agg(
            {"morph_code": "mean", "Hour_GBs": "sum"}
        )
        sinr_bins_agg["xnpv"] = sinr_bins_agg.apply(
            lambda x: loop_npv29(x["Hour_GBs"], x["morph_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = sinr_bins_agg["xnpv"].min()
        next_best["sum_npv"] = sinr_bins_agg["xnpv"].sum()
        next_best = next_best[["fict_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= npv_threshold:
            bad_candidate = next_best["fict_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fict_site")
        init_cand_bins = init_cand_bins.nlargest(bin_prox, "sinr")
        init_cand_bins = init_cand_bins[["sinr", "fict_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_sinr_bins, on="fict_site")
        cand_bins = cand_bins.nlargest(bin_prox, "sinr_new")
        cand_bins = cand_bins[["sinr_new"]].mean()
        if (abs(init_cand_bins["sinr"] - cand_bins["sinr_new"])) > sinr_deg:
            bad_candidate = init_cand_bins["fict_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fict_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fict_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fict_site"})

pole_du = pole_du[1:11, :]
pole_u = pole_u[1:11, :]
pole_s = pole_s[1:11, :]

start_yr10 = timeit.default_timer()
sites_yr10 = sites_yr10[["fict_site"]]
sites_10 = pd.merge(calc_sites, sites_yr10, on="fict_site")

if len(sites_yr10) > 0:
    sites = sites_10
    sites = sites.groupby("fict_site", as_index=False).agg(
        {"morph_code": "mean", "Hour_GBs": "sum"}
    )
    sites["npv"] = sites.apply(
        lambda x: loop_npv30(x["Hour_GBs"], x["morph_code"]), axis=1
    ).astype(int)
    ranking = sites.loc[sites["npv"] > 0].copy()
    ranking["rank"] = ranking["npv"].rank(ascending=False)
    ranking = ranking.sort_values(by="rank")
    temp_ranking = ranking[["fict_site"]]
    init_ranking_bins = pd.merge(temp_ranking, sites_10, on="fict_site")
    init_ranking_counts = init_ranking_bins["GridName"].value_counts().reset_index()
    init_ranking_counts = init_ranking_counts.rename(
        columns={"index": "GridName", "GridName": "bin_count"}
    )
    init_ranking = pd.merge(init_ranking_bins, init_ranking_counts, on="GridName")
    if len(selected) > 0 and len(ranking) > 0:
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
        new_selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected.append(new_selected, sort=True)
        selected = selected[["fict_site", "npv"]]
    elif len(selected) == 0 and len(ranking) > 0:
        selected = (pd.DataFrame(ranking.iloc[0, :])).T
        selected = selected[["fict_site", "npv"]]
        candidates = pd.DataFrame(ranking.iloc[1:, :]).reset_index(drop=True)
        candidates = candidates[["fict_site", "npv"]]
    else:
        candidates = pd.DataFrame([])

for j in range(len(candidates)):
    temp_nb = pd.DataFrame([])
    for i in range(len(candidates)):
        next_best = (pd.DataFrame(candidates.iloc[i, :])).T
        sinr_sites = selected.append(next_best, sort=True)
        sinr_bins = pd.merge(init_ranking, sinr_sites, on="fict_site")
        sinr_bins_unique = sinr_bins[sinr_bins["bin_count"] == 1].copy()
        sinr_bins_dups = sinr_bins[sinr_bins["bin_count"] > 1].copy()
        temp_sinr_bins_dups = (
            sinr_bins_dups.groupby("GridName")["rx_signal_strength_mw"]
                .sum()
                .reset_index()
        )
        sinr_bins_dups = sinr_bins_dups.drop(["rx_signal_strength_mw"], axis=1)
        sinr_bins_dups = pd.merge(sinr_bins_dups, temp_sinr_bins_dups, on="GridName")
        sinr_bins_unique["sinr_new"] = sinr_bins_unique["sinr"]
        sinr_bins_dups["sinr_new"] = sinr_new(
            sinr_bins_dups["rx_signal_strength_mw"],
            sinr_bins_dups["rx_signal_strength_db"],
        )
        sinr_bins = sinr_bins_unique.append(sinr_bins_dups, sort=True)
        temp_sinr_bins = sinr_bins
        sinr_bins = sinr_bins[sinr_bins.sinr_new >= -7.0]
        sinr_bins_agg = sinr_bins.groupby("fict_site", as_index=False).agg(
            {"morph_code": "mean", "Hour_GBs": "sum"}
        )
        sinr_bins_agg["xnpv"] = sinr_bins_agg.apply(
            lambda x: loop_npv30(x["Hour_GBs"], x["morph_code"]), axis=1
        ).astype(int)
        next_best["adj_npv"] = sinr_bins_agg["xnpv"].min()
        next_best["sum_npv"] = sinr_bins_agg["xnpv"].sum()
        next_best = next_best[["fict_site", "adj_npv", "sum_npv"]]
        if next_best["adj_npv"].values <= npv_threshold:
            bad_candidate = next_best["fict_site"].values
            full = 0
            break
        init_cand_bins = pd.merge(next_best, init_ranking, on="fict_site")
        init_cand_bins = init_cand_bins.nlargest(bin_prox, "sinr")
        init_cand_bins = init_cand_bins[["sinr", "fict_site"]].mean().astype("int64")
        cand_bins = pd.merge(next_best, temp_sinr_bins, on="fict_site")
        cand_bins = cand_bins.nlargest(bin_prox, "sinr_new")
        cand_bins = cand_bins[["sinr_new"]].mean()
        if (abs(init_cand_bins["sinr"] - cand_bins["sinr_new"])) > sinr_deg:
            bad_candidate = init_cand_bins["fict_site"].astype("int64")
            full = 0
            break
        temp_nb = temp_nb.append(next_best, sort=True)
        full = 1
    if full == 1:
        if len(candidates) == 0:
            break
        new_ranking = temp_nb
        new_ranking["rank"] = new_ranking["sum_npv"].rank(ascending=True)
        new_ranking = new_ranking.sort_values(by="rank").reset_index(drop="True")
        temp_selected = new_ranking[new_ranking["rank"] == 1]
        temp_selected = temp_selected[["fict_site"]]
        selected = selected.append(temp_selected, sort=True)
        new_ranking = new_ranking[["fict_site"]]
        candidates = pd.DataFrame(new_ranking.iloc[1:, :]).reset_index(drop=True)
    elif full == 0:
        if len(candidates) == 0:
            break
        candidates = candidates.to_numpy()
        candidates = np.delete(
            candidates, np.where(candidates == bad_candidate)[0], axis=0
        )
        candidates = pd.DataFrame(candidates).reset_index(drop=True)
        candidates = candidates.rename(columns={0: "fict_site"})

pole_du = fin_arrays_pole(1)
pole_u = fin_arrays_pole(2)
pole_s = fin_arrays_pole(3)

selected = pd.merge(selected, init_build_yr_sites, on="fict_site")
selected = pd.merge(selected, init_sites, on="fict_site")

site_bin_gbs = selected.groupby("fict_site")["Hour_GBs"].sum().reset_index()
bld_yr_sites = pd.merge(site_bin_gbs, site_t_m, on="fict_site")

bld_yr_sites["1"] = bld_yr_sites.apply(
    lambda x: bld_npv21(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["2"] = bld_yr_sites.apply(
    lambda x: bld_npv22(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["3"] = bld_yr_sites.apply(
    lambda x: bld_npv23(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["4"] = bld_yr_sites.apply(
    lambda x: bld_npv24(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["5"] = bld_yr_sites.apply(
    lambda x: bld_npv25(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["6"] = bld_yr_sites.apply(
    lambda x: bld_npv26(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["7"] = bld_yr_sites.apply(
    lambda x: bld_npv27(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["8"] = bld_yr_sites.apply(
    lambda x: bld_npv28(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["9"] = bld_yr_sites.apply(
    lambda x: bld_npv29(x["Hour_GBs"], x["morph_code"]), axis=1
)
bld_yr_sites["10"] = bld_yr_sites.apply(
    lambda x: bld_npv30(x["Hour_GBs"], x["morph_code"]), axis=1
)

inh_bld_yr = bld_yr_sites[["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]]

bld_yr_sites["design_build_yr"] = inh_bld_yr.idxmin(axis=1).astype(int)
final_build_yr_sites = bld_yr_sites[["fict_site", "design_build_yr"]]
selected = pd.merge(selected, final_build_yr_sites, on="fict_site")
selected["opt_build_year"] = (
        (selected["design_build_yr"] + selected["build_yr"]) / 2
).round(0)

selected = selected.drop(["build_yr", "design_build_yr"], axis=1)
selected_bins = selected["GridName"].value_counts().reset_index()
selected_bins = selected_bins.rename(
    columns={"index": "GridName", "GridName": "bin_count"}
)
selected = pd.merge(selected, selected_bins, on="GridName")

selected_unique = selected[selected["bin_count"] == 1].copy()
selected_dups = selected[selected["bin_count"] > 1].copy()
temp_selected_dups = (
    selected_dups.groupby("GridName")["rx_signal_strength_mw"].sum().reset_index()
)
selected_dups = selected_dups.drop(["rx_signal_strength_mw"], axis=1)
selected_dups = pd.merge(selected_dups, temp_selected_dups, on="GridName")
selected_unique["sinr_new"] = selected_unique["sinr"]
selected_dups["sinr_new"] = sinr_new(
    selected_dups["rx_signal_strength_mw"], selected_dups["rx_signal_strength_db"]
)
selected = selected_unique.append(selected_dups, sort=True)

selected = selected[selected.sinr_new >= -7.0]

end_time = timeit.default_timer()
temp_selected = selected.groupby("fict_site")["morph_code"].first().reset_index()
print(temp_selected["morph_code"].value_counts())

print("final_sites:", end=" ")
print(len(selected["fict_site"].unique()))

print("site_count:", end=" ")
print(site_number)

print("total_sec:", end=" ")
print(end_time - init_time)

print("total_min:", end=" ")
print((end_time - init_time) / 60)

print("sec/site:", end=" ")
print((end_time - init_time) / site_number)

print("mean_sinr", end=" ")
print(selected["sinr_new"].mean())

# NPV DATA

npv_values.columns = ['non', 'pole_du', 'pole_u', 'pole_s', '', 'off_du', 'off_u',
                      'off_s', 'off_r', 'roe_du', 'roe_u', 'roe_s', 'roe_r',
                      'smb_du', 'smb_u', 'smb_s', 'smb_r']

# This creates time periods for npv
st_date20 = pd.date_range(start='2020-06-30', periods=11, freq='12M')
st_date21 = pd.date_range(start='2021-06-30', periods=10, freq='12M')
st_date22 = pd.date_range(start='2022-06-30', periods=9, freq='12M')
st_date23 = pd.date_range(start='2023-06-30', periods=8, freq='12M')
st_date24 = pd.date_range(start='2024-06-30', periods=7, freq='12M')
st_date25 = pd.date_range(start='2025-06-30', periods=6, freq='12M')
st_date26 = pd.date_range(start='2026-06-30', periods=5, freq='12M')
st_date27 = pd.date_range(start='2027-06-30', periods=4, freq='12M')
st_date28 = pd.date_range(start='2028-06-30', periods=3, freq='12M')
st_date29 = pd.date_range(start='2029-06-30', periods=2, freq='12M')
st_date30 = pd.date_range(start='2030-06-30', periods=1, freq='12M')

disc_rt = .15

# LTE INPUTS

rx_sensitivity_db = -89.4


# SECTION 2: FUNCTIONS

def fin_arrays(code):
    global npv
    cpx = np.empty([11, 1])
    opx = np.empty([11, 1])
    growth = np.empty([11, 1])
    mvno = np.empty([11, 1])
    if code == 1:
        npv = npv_values.loc[:, 'pole_du']
    elif code == 2:
        npv = npv_values.loc[:, 'pole_u']
    elif code == 3:
        npv = npv_values.loc[:, 'pole_s']
    elif code == 4:
        npv = npv_values.loc[:, '']
    elif code == 5:
        npv = npv_values.loc[:, 'off_du']
    elif code == 6:
        npv = npv_values.loc[:, 'off_u']
    elif code == 7:
        npv = npv_values.loc[:, 'off_s']
    elif code == 8:
        npv = npv_values.loc[:, 'off_r']
    elif code == 9:
        npv = npv_values.loc[:, 'roe_du']
    elif code == 10:
        npv = npv_values.loc[:, 'roe_u']
    elif code == 11:
        npv = npv_values.loc[:, 'roe_s']
    elif code == 12:
        npv = npv_values.loc[:, 'roe_r']
    elif code == 13:
        npv = npv_values.loc[:, 'smb_du']
    elif code == 14:
        npv = npv_values.loc[:, 'smb_u']
    elif code == 15:
        npv = npv_values.loc[:, 'smb_s']
    elif code == 16:
        npv = npv_values.loc[:, 'smb_r']
    for i in range(11):
        cpx[i] = float(npv[i + 1])
    for i in range(11):
        opx[i] = float(npv[i + 12])
    for i in range(11):
        growth[i] = float(npv[i + 23])
    for i in range(11):
        mvno[i] = float(npv[i + 34])
    array = np.hstack((cpx, opx, growth, mvno))
    return array


def morph_array(code):
    if code == 1:
        return pole_du
    elif code == 2:
        return pole_u
    elif code == 3:
        return pole_s


day_diff = 0.0
cell_split = 2.0


def capex_21(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[1:11, :]
    debit = 0
    for i in range(10):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = array[i][0]
            return debit


def capex_22(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[2:11, :]
    for i in range(9):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = array[i][0]
            return debit


def capex_23(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[3:11, :]
    debit = 0
    for i in range(8):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = array[i][0]
            return debit


def capex_24(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[4:11, :]
    debit = 0
    for i in range(7):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = array[i][0]
            return debit


def capex_25(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[5:11, :]
    debit = 0
    for i in range(6):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = array[i][0]
            return debit


def capex_26(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[6:11, :]
    debit = 0
    for i in range(5):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = array[i][0]
            return debit


def capex_27(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[7:11, :]
    debit = 0
    for i in range(4):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = array[i][0]
            return debit


def capex_28(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[8:11, :]
    debit = 0
    for i in range(3):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = array[i][0]
            return debit


def capex_29(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[9:11, :]
    debit = 0
    for i in range(2):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = array[i][0]
            return debit


def capex_30(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[10:11, :]
    debit = 0
    for i in range(10):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = array[i][0]
            return debit


def opex_21(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[1:11, :]
    debit = 0
    for i in range(10):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = debit + array[i][1]
    return debit


def opex_22(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[2:11, :]
    debit = 0
    for i in range(9):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = debit + array[i][1]
    return debit


def opex_23(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[3:11, :]
    debit = 0
    for i in range(8):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = debit + array[i][1]
    return debit


def opex_24(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[4:11, :]
    debit = 0
    for i in range(7):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = debit + array[i][1]
    return debit


def opex_25(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[5:11, :]
    debit = 0
    for i in range(6):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = debit + array[i][1]
    return debit


def opex_26(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[6:11, :]
    debit = 0
    for i in range(5):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = debit + array[i][1]
    return debit


def opex_27(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[7:11, :]
    debit = 0
    for i in range(4):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = debit + array[i][1]
    return debit


def opex_28(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[8:11, :]
    debit = 0
    for i in range(3):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = debit + array[i][1]
    return debit


def opex_29(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[9:11, :]
    debit = 0
    for i in range(2):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = debit + array[i][1]
    return debit


def opex_30(gbs, mo_cap, code):
    array = morph_array(code)
    array = array[10:11, :]
    debit = 0
    for i in range(1):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            debit = debit + array[i][1]
    return debit


def split21(gbs, mo_cap, code):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    array = morph_array(code)
    array = array[1:11, :]
    year = 1
    for i in range(10):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            return year
        else:
            year = year + 1


def split22(gbs, mo_cap, code):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([9, 1])
    array = morph_array(code)
    array = array[2:11, :]
    year = 2
    for i in range(9):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            return year
        else:
            year = year + 1


def split23(gbs, mo_cap, code):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([8, 1])
    array = morph_array(code)
    array = array[3:11, :]
    year = 3
    for i in range(8):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            return year
        else:
            year = year + 1


def split24(gbs, mo_cap, code):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([7, 1])
    array = morph_array(code)
    array = array[4:11, :]
    year = 4
    for i in range(7):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            return year
        else:
            year = year + 1


def split25(gbs, mo_cap, code):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([6, 1])
    array = morph_array(code)
    array = array[5:11, :]
    year = 5
    for i in range(6):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            return year
        else:
            year = year + 1


def split26(gbs, mo_cap, code):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([5, 1])
    array = morph_array(code)
    array = array[6:11, :]
    year = 6
    for i in range(5):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            return year
        else:
            year = year + 1


def split27(gbs, mo_cap, code):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([4, 1])
    array = morph_array(code)
    array = array[7:11, :]
    year = 7
    for i in range(4):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            return year
        else:
            year = year + 1


def split28(gbs, mo_cap, code):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([3, 1])
    array = morph_array(code)
    array = array[8:11, :]
    year = 8
    for i in range(3):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            return year
        else:
            year = year + 1


def split29(gbs, mo_cap, code):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([2, 1])
    array = morph_array(code)
    array = array[9:11, :]
    year = 9
    for i in range(2):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            return year
        else:
            year = year + 1


def split30(gbs, mo_cap, code):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([1, 1])
    array = morph_array(code)
    array = array[10:11, :]
    year = 10
    for i in range(1):
        if (gbs * (array[i][2]) * 365) >= (mo_cap * 365):
            return year
        else:
            year = year + 1


def npv21(gbs, code, mo_cap):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([10, 1])
    array = morph_array(code)
    array = array[1:11, :]
    for i in range(10):
        if (gbs * (array[i][2]) * 365) < (cell_split * mo_cap * 365):
            value[i] = gbs * 365 * array[i][3] * array[i][2] - array[i][1]
        else:
            value[i] = cell_split * mo_cap * 365
    date_0 = st_date20[0]
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** ((date_i - date_0).days / 365.0))
                    for value_i, date_i in zip(value, st_date21)]))


def npv22(gbs, code, mo_cap):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([9, 1])
    array = morph_array(code)
    array = array[2:11, :]
    a = 0
    for i in range(9):
        if (gbs * (array[i][2]) * 365) < (cell_split * mo_cap * 365):
            value[i] = gbs * 365 * array[i][3] * array[i][2] - array[i][1]
        else:
            value[i] = cell_split * mo_cap * 365
    date_0 = st_date20[0]

    return int(
        sum([value_i / ((1.0 + disc_rt) ** (((date_i - date_0).days - day_diff) / 365.0))
             for value_i, date_i in zip(value, st_date22)]))


def npv23(gbs, code, mo_cap):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([8, 1])
    array = morph_array(code)
    array = array[3:11, :]
    a = 0
    for i in range(8):
        if (gbs * (array[i][2]) * 365) < (cell_split * mo_cap * 365):
            value[i] = gbs * 365 * array[i][3] * array[i][2] - array[i][1]
        else:
            value[i] = cell_split * mo_cap * 365
    date_0 = st_date20[0]
    value[a] = value[a] - array[a][0]
    return int(
        sum([value_i / ((1.0 + disc_rt) ** (((date_i - date_0).days - day_diff) / 365.0))
             for value_i, date_i in zip(value, st_date23)]))


def npv24(gbs, code, mo_cap):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([7, 1])
    array = morph_array(code)
    array = array[4:11, :]
    a = 0
    for i in range(7):
        if (gbs * (array[i][2]) * 365) < (cell_split * mo_cap * 365):
            value[i] = gbs * 365 * array[i][3] * array[i][2] - array[i][1]
        else:
            value[i] = cell_split * mo_cap * 365
    date_0 = st_date20[0]
    value[a] = value[a] - array[a][0]
    return int(
        sum([value_i / ((1.0 + disc_rt) ** (((date_i - date_0).days - day_diff) / 365.0))
             for value_i, date_i in zip(value, st_date24)]))


def npv25(gbs, code, mo_cap):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([6, 1])
    array = morph_array(code)
    array = array[5:11, :]
    a = 0
    for i in range(6):
        if (gbs * (array[i][2]) * 365) < (cell_split * mo_cap * 365):
            value[i] = gbs * 365 * array[i][3] * array[i][2] - array[i][1]
        else:
            value[i] = cell_split * mo_cap * 365
    date_0 = st_date20[0]
    value[a] = value[a] - array[a][0]
    return int(
        sum([value_i / ((1.0 + disc_rt) ** (((date_i - date_0).days - day_diff) / 365.0))
             for value_i, date_i in zip(value, st_date25)]))


def npv26(gbs, code, mo_cap):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([5, 1])
    array = morph_array(code)
    array = array[6:11, :]
    a = 0
    for i in range(5):
        if (gbs * (array[i][2]) * 365) < (cell_split * mo_cap * 365):
            value[i] = gbs * 365 * array[i][3] * array[i][2] - array[i][1]
        else:
            value[i] = cell_split * mo_cap * 365
    date_0 = st_date20[0]
    value[a] = value[a] - array[a][0]
    return int(
        sum([value_i / ((1.0 + disc_rt) ** (((date_i - date_0).days - day_diff) / 365.0))
             for value_i, date_i in zip(value, st_date26)]))


def npv27(gbs, code, mo_cap):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([4, 1])
    array = morph_array(code)
    array = array[7:11, :]
    a = 0
    for i in range(4):
        if (gbs * (array[i][2]) * 365) < (cell_split * mo_cap * 365):
            value[i] = gbs * 365 * array[i][3] * array[i][2] - array[i][1]
        else:
            value[i] = cell_split * mo_cap * 365
    date_0 = st_date20[0]
    value[a] = value[a] - array[a][0]
    return int(
        sum([value_i / ((1.0 + disc_rt) ** (((date_i - date_0).days - day_diff) / 365.0))
             for value_i, date_i in zip(value, st_date27)]))


def npv28(gbs, code, mo_cap):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([3, 1])
    array = morph_array(code)
    array = array[8:11, :]
    for i in range(3):
        if (gbs * (array[i][2]) * 365) < (cell_split * mo_cap * 365):
            value[i] = gbs * 365 * array[i][3] * array[i][2] - array[i][1]
        else:
            value[i] = cell_split * mo_cap * 365
    date_0 = st_date20[0]
    value[0] = value[0] - array[0][0]
    return int(sum([value_i / ((1.0 + disc_rt) ** ((date_i - date_0).days / 365.0))
                    for value_i, date_i in zip(value, st_date28)]))


def npv29(gbs, code, mo_cap):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([2, 1])
    array = morph_array(code)
    array = array[9:11, :]
    a = 0
    for i in range(2):
        if (gbs * (array[i][2]) * 365) < (cell_split * mo_cap * 365):
            value[i] = gbs * 365 * array[i][3] * array[i][2] - array[i][1]
        else:
            value[i] = cell_split * mo_cap * 365
    date_0 = st_date20[0]
    value[a] = value[a] - array[a][0]
    return int(
        sum([value_i / ((1.0 + disc_rt) ** (((date_i - date_0).days - day_diff) / 365.0))
             for value_i, date_i in zip(value, st_date29)]))


def npv30(gbs, code, mo_cap):
    # slice 0 is cpx, 1 is opx, 2 is growth, 3 is mvno,
    value = np.empty([1, 1])
    array = morph_array(code)
    array = array[10:11, :]
    a = 0
    for i in range(1):
        if (gbs * (array[i][2]) * 365) < (cell_split * mo_cap * 365):
            value[i] = gbs * 365 * array[i][3] * array[i][2] - array[i][1]
        else:
            value[i] = cell_split * mo_cap * 365
    date_0 = st_date20[0]
    value[a] = value[a] - array[a][0]
    return int(
        sum([value_i / ((1.0 + disc_rt) ** (((date_i - date_0).days - day_diff) / 365.0))
             for value_i, date_i in zip(value, st_date30)]))


def extra_offload21(gbs, code, mo_cap):
    cap = np.empty([10, 1])
    array = morph_array(code)
    array = array[1:11, :]
    for i in range(10):
        if (gbs * (array[i][2]) * 365) < (mo_cap * 365):
            cap[i] = (mo_cap * 365) - (gbs * (array[i][2]) * 365)
        else:
            cap[i] = 0
    return cap


def extra_offload22(gbs, code, mo_cap):
    cap = np.empty([9, 1])
    array = morph_array(code)
    array = array[2:11, :]
    for i in range(9):
        if (gbs * (array[i][2]) * 365) < (mo_cap * 365):
            cap[i] = (mo_cap * 365) - (gbs * (array[i][2]) * 365)
        else:
            cap[i] = 0
    return cap


def extra_offload23(gbs, code, mo_cap):
    cap = np.empty([8, 1])
    array = morph_array(code)
    array = array[3:11, :]
    for i in range(8):
        if (gbs * (array[i][2]) * 365) < (mo_cap * 365):
            cap[i] = (mo_cap * 365) - (gbs * (array[i][2]) * 365)
        else:
            cap[i] = 0
    return cap


def extra_offload24(gbs, code, mo_cap):
    cap = np.empty([7, 1])
    array = morph_array(code)
    array = array[4:11, :]
    for i in range(7):
        if (gbs * (array[i][2]) * 365) < (mo_cap * 365):
            cap[i] = (mo_cap * 365) - (gbs * (array[i][2]) * 365)
        else:
            cap[i] = 0
    return cap


def extra_offload25(gbs, code, mo_cap):
    cap = np.empty([6, 1])
    array = morph_array(code)
    array = array[5:11, :]
    for i in range(6):
        if (gbs * (array[i][2]) * 365) < (mo_cap * 365):
            cap[i] = (mo_cap * 365) - (gbs * (array[i][2]) * 365)
        else:
            cap[i] = 0
    return cap


def extra_offload26(gbs, code, mo_cap):
    cap = np.empty([5, 1])
    array = morph_array(code)
    array = array[6:11, :]
    for i in range(5):
        if (gbs * (array[i][2]) * 365) < (mo_cap * 365):
            cap[i] = (mo_cap * 365) - (gbs * (array[i][2]) * 365)
        else:
            cap[i] = 0
    return cap


def extra_offload27(gbs, code, mo_cap):
    cap = np.empty([4, 1])
    array = morph_array(code)
    array = array[7:11, :]
    for i in range(4):
        if (gbs * (array[i][2]) * 365) < (mo_cap * 365):
            cap[i] = (mo_cap * 365) - (gbs * (array[i][2]) * 365)
        else:
            cap[i] = 0
    return cap


def extra_offload28(gbs, code, mo_cap):
    cap = np.empty([3, 1])
    array = morph_array(code)
    array = array[8:11, :]
    for i in range(3):
        if (gbs * (array[i][2]) * 365) < (mo_cap * 365):
            cap[i] = (mo_cap * 365) - (gbs * (array[i][2]) * 365)
        else:
            cap[i] = 0
    return cap


def extra_offload29(gbs, code, mo_cap):
    cap = np.empty([2, 1])
    array = morph_array(code)
    array = array[9:11, :]
    for i in range(2):
        if (gbs * (array[i][2]) * 365) < (mo_cap * 365):
            cap[i] = (mo_cap * 365) - (gbs * (array[i][2]) * 365)
        else:
            cap[i] = 0
    return cap


def extra_offload30(gbs, code, mo_cap):
    cap = np.empty([1, 1])
    array = morph_array(code)
    array = array[10:11, :]
    for i in range(1):
        if (gbs * (array[i][2]) * 365) < (mo_cap * 365):
            cap[i] = (mo_cap * 365) - (gbs * (array[i][2]) * 365)
        else:
            cap[i] = 0
    return cap


def rb_thru_put(code_rate, symbols, mimo, subframe, retrans, high_layer_over,
                over_tp_kbps):
    rb_thru = (((code_rate * symbols * 4800000 * mimo * subframe) / 1000000) *
               ((1 - retrans) * (1 - high_layer_over)) - (over_tp_kbps / 1000)) / 400
    return rb_thru


def rx_calc(path_loss_umi_db):
    rx_signal_strength_db = 37.0 - path_loss_umi_db
    sinr = (rx_signal_strength_db - rx_sensitivity_db).round(0).astype(int)
    rx_signal_strength_mw = (10 ** ((37.0 - path_loss_umi_db) / 10) / 1000.000000000)
    return rx_signal_strength_db, sinr, rx_signal_strength_mw


def sinr_new(sum_rx_signal, rx_signal_strength_db):
    # note that -89.4 here is rx_sensitivity_db
    sinr_nouveau = np.round(
        sum_rx_signal.where(sum_rx_signal.isnull(), rx_signal_strength_db -
                            (np.log10((sum_rx_signal + (
                                        10 ** (-89.4 / 10)) / 1000) * 1000) * 10)),
        0)
    return sinr_nouveau


def sinr_new2(sum_rx_signal, rx_signal_strength_db):
    # note that -89.4 here is rx_sensitivity_db
    sinr_nu = np.round(sum_rx_signal.where(sum_rx_signal.isnull(), rx_signal_strength_db -
                                           (np.log10((sum_rx_signal + (10 ** (
                                                       -89.4 / 10)) / 1000) * 1000) * 10)),
                       0)
    return sinr_nu


def reverse_morph_type(code):
    global type, morph, type_morph
    if code == 1:
        type = 'Pole'
        morph = 'Dense Urban'
        type_morph = 'Pole/Dense Urban'
    elif code == 2:
        type = 'Pole'
        morph = 'Urban'
        type_morph = 'Pole/Urban'
    elif code == 3:
        type = 'Pole'
        morph = 'Suburban'
        type_morph = 'Pole / Suburban'
    return type


def reverse_morph_morph(code):
    global type, morph, type_morph
    if code == 1:
        type = 'Pole'
        morph = 'Dense Urban'
        type_morph = 'Pole/Dense Urban'
    elif code == 2:
        type = 'Pole'
        morph = 'Urban'
        type_morph = 'Pole/Urban'
    elif code == 3:
        type = 'Pole'
        morph = 'Suburban'
        type_morph = 'Pole / Suburban'
    return morph


def reverse_morph_type_morph(code):
    global type, morph, type_morph
    if code == 1:
        type = 'Pole'
        morph = 'Dense Urban'
        type_morph = 'Pole/Dense Urban'
    elif code == 2:
        type = 'Pole'
        morph = 'Urban'
        type_morph = 'Pole/Urban'
    elif code == 3:
        type = 'Pole'
        morph = 'Suburban'
        type_morph = 'Pole / Suburban'
    return type_morph


pole_du = fin_arrays(1)
pole_u = fin_arrays(2)
pole_s = fin_arrays(3)
off_du = fin_arrays(5)
off_u = fin_arrays(6)
off_s = fin_arrays(7)
off_r = fin_arrays(8)
roe_du = fin_arrays(9)
roe_u = fin_arrays(10)
roe_s = fin_arrays(11)
roe_r = fin_arrays(12)
smb_du = fin_arrays(13)
smb_u = fin_arrays(14)
smb_s = fin_arrays(15)
smb_r = fin_arrays(16)

all_segs_init = selected

all_segs = all_segs_init

all_segs['rx_signal_strength_db'], all_segs['sinr'], all_segs['rx_signal_strength_mw'] = \
    rx_calc(all_segs['path_loss_umi_db'])
all_segs.loc[all_segs['sinr'] > 50, 'sinr'] = 50.0

sinr_bins_unique = all_segs.drop_duplicates(subset='GridName', keep=False).copy()
sinr_bins_dups = all_segs[all_segs.duplicated(['GridName'], keep=False)].copy()
sinr_bins_dups['sum_rx_signal'] = sinr_bins_dups.groupby('GridName')[
    'rx_signal_strength_mw'].transform(sum)
all_segs = sinr_bins_unique.append(sinr_bins_dups)
sinr_bins_a = all_segs[all_segs['sum_rx_signal'].notnull()].copy().reset_index(drop=True)

sinr_bins_a['sinr_new'] = sinr_new2(sinr_bins_a['sum_rx_signal'],
                                    sinr_bins_a['rx_signal_strength_db'])

sinr_bins_b = all_segs[all_segs['sum_rx_signal'].isnull()].copy()
sinr_bins_b['sinr_new'] = sinr_bins_b['sum_rx_signal'].where(
    sinr_bins_b['sum_rx_signal'].notnull(),
    sinr_bins_b['sinr'], axis=0)
all_segs = sinr_bins_a.append(sinr_bins_b)
all_segs = all_segs[all_segs.sinr_new >= -2.0]
all_segs['gb_offload'] = all_segs.groupby('fict_site')['Hour_GBs'].transform(sum)
all_segs = all_segs.drop(['rx_signal_strength_db', 'rx_signal_strength_mw', 'sinr',
                          'sum_rx_signal', 'gb_offload'], axis=1)
all_segs['grid_temp'] = all_segs['GridName'].str.replace(r'\D', '')
all_segs['grid_temp'] = all_segs['grid_temp'].str[-8:]
all_segs = all_segs.sort_values(['grid_temp', 'sinr_new', 'opt_build_year'])

all_segs_unique = all_segs.drop_duplicates(subset='grid_temp', keep=False).copy()
all_segs_dups = all_segs[all_segs.duplicated(['grid_temp'], keep=False)].copy()
all_segs = all_segs_unique.append(all_segs_dups)

all_segs = pd.merge(all_segs, lte_params, left_on='sinr_new', right_on='SINR')
all_segs['rb_thru_put'] = rb_thru_put(all_segs['Code Rate'], all_segs['symbols/SF'],
                                      all_segs['2x2 MIMO Gain'],
                                      all_segs['subframe allocation'],
                                      all_segs['retrans'],
                                      all_segs['high  layer overhead'],
                                      all_segs['overhead TP kbps'])
all_segs['gb_offload'] = all_segs.groupby('fict_site')['Hour_GBs'].transform(sum)
print(lte_params['Code Rate'].mean())
all_segs['rx_signal_strength_db'], all_segs['sinr'], all_segs['rx_signal_strength_mw'] = \
    rx_calc(all_segs['path_loss_umi_db'])
all_segs.loc[all_segs['sinr'] > 50, 'sinr'] = 50.0

fict_site_count_a = all_segs[['GridName', 'fict_site']]

fict_site_count = fict_site_count_a['fict_site'].value_counts().reset_index()
fict_site_count = fict_site_count.rename(
    columns={'index': 'fict_site', 'fict_site': 'fict_site_count'})
all_segs = pd.merge(all_segs, fict_site_count, on='fict_site')

all_segs = all_segs.loc[all_segs['fict_site_count'] > 61]

sinr_bins_unique = all_segs.drop_duplicates(subset='GridName', keep=False).copy()
sinr_bins_dups = all_segs[all_segs.duplicated(['GridName'], keep=False)].copy()
sinr_bins_dups['sum_rx_signal'] = sinr_bins_dups.groupby('GridName')[
    'rx_signal_strength_mw'].transform(sum)
all_segs = sinr_bins_unique.append(sinr_bins_dups)
sinr_bins_a = all_segs[all_segs['sum_rx_signal'].notnull()].copy().reset_index(drop=True)

sinr_bins_a['sinr_new'] = sinr_new2(sinr_bins_a['sum_rx_signal'],
                                    sinr_bins_a['rx_signal_strength_db'])

sinr_bins_b = all_segs[all_segs['sum_rx_signal'].isnull()].copy()
sinr_bins_b['sinr_new'] = sinr_bins_b['sum_rx_signal'].where(
    sinr_bins_b['sum_rx_signal'].notnull(),
    sinr_bins_b['sinr'], axis=0)
all_segs = sinr_bins_a.append(sinr_bins_b)
all_segs = all_segs[all_segs.sinr_new >= -2.0]
all_segs['gb_offload'] = all_segs.groupby('fict_site')['Hour_GBs'].transform(sum)
all_segs = all_segs.drop(['path_loss_umi_db', 'Code Rate', 'symbols/SF', '2x2 MIMO Gain',
                          'rx_signal_strength_db', 'rx_signal_strength_mw', 'sinr',
                          'sum_rx_signal', 'gb_offload', 'subframe allocation',
                          'retrans', 'high  layer overhead', 'overhead TP kbps'], axis=1)

all_segs['grid_temp'] = all_segs['GridName'].str.replace(r'\D', '')
all_segs['grid_temp'] = all_segs['grid_temp'].str[-8:]
all_segs = all_segs.sort_values(['grid_temp', 'sinr_new', 'opt_build_year'])

all_segs_unique = all_segs.drop_duplicates(subset='grid_temp', keep=False).copy()
all_segs_dups = all_segs[all_segs.duplicated(['grid_temp'], keep=False)].copy()
all_segs = all_segs_unique.append(all_segs_dups)

all_segs = pd.merge(all_segs, lte_params, left_on='sinr_new', right_on='SINR')
all_segs['rb_thru_put'] = rb_thru_put(all_segs['Code Rate'], all_segs['symbols/SF'],
                                      all_segs['2x2 MIMO Gain'],
                                      all_segs['subframe allocation'],
                                      all_segs['retrans'],
                                      all_segs['high  layer overhead'],
                                      all_segs['overhead TP kbps'])
all_segs['gb_offload'] = all_segs.groupby('fict_site')['Hour_GBs'].transform(sum)
test = all_segs[['GridName', 'fict_site']]

fict_site_count_2 = test['fict_site'].value_counts().reset_index()
fict_site_count_2 = fict_site_count_2.rename(
    columns={'index': 'fict_site', 'fict_site': 'fict_site_count_2'})
fict_site_check = pd.merge(fict_site_count, fict_site_count_2, on='fict_site')
fict_site_check['diff'] = fict_site_check['fict_site_count'] - fict_site_check[
    'fict_site_count_2']

all_segs = pd.merge(all_segs, fict_site_count_2, on='fict_site')

all_segs = all_segs.sort_values(by=['fict_site'], ascending=True)

all_segs = pd.merge(all_segs, bin_req, on='GridName')

sinr_test = all_segs
all_segs['bh_req_rbs'] = all_segs['bin_req_hr_mbps'] / all_segs['rb_thru_put']
all_segs = all_segs[
    ['fict_site', 'morph_code', 'gb_offload', 'bh_req_rbs', 'opt_build_year']]
all_segs = all_segs.groupby('fict_site', as_index=False).agg(
    {'bh_req_rbs': sum, 'morph_code': 'mean', 'gb_offload': 'mean',
     'opt_build_year': 'mean'})
all_segs['enb_util'] = np.ceil(all_segs['bh_req_rbs']) / 400
all_segs['max_cap'] = (all_segs['gb_offload'] / all_segs['enb_util'])
all_segs = pd.merge(all_segs, fict_site_count_2, on='fict_site')

npv2021 = all_segs[all_segs['opt_build_year'] == 1].copy().reset_index()
npv2022 = all_segs[all_segs['opt_build_year'] == 2].copy().reset_index()
npv2023 = all_segs[all_segs['opt_build_year'] == 3].copy().reset_index()
npv2024 = all_segs[all_segs['opt_build_year'] == 4].copy().reset_index()
npv2025 = all_segs[all_segs['opt_build_year'] == 5].copy().reset_index()
npv2026 = all_segs[all_segs['opt_build_year'] == 6].copy().reset_index()
npv2027 = all_segs[all_segs['opt_build_year'] == 7].copy().reset_index()
npv2028 = all_segs[all_segs['opt_build_year'] == 8].copy().reset_index()
npv2029 = all_segs[all_segs['opt_build_year'] == 9].copy().reset_index()
npv2030 = all_segs[all_segs['opt_build_year'] == 10].copy().reset_index()

if len(npv2021) > 0:
    npv2021['npv'] = npv2021.apply(
        lambda x: npv21(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).astype(int)
    npv2021['split_yr'] = npv2021.apply(
        lambda x: split21(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1)
    npv2021['debit'] = npv2021.apply(
        lambda x: capex_21(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1) + \
                       npv2021.apply(lambda x: opex_21(x['gb_offload'], x['max_cap'],
                                                       x['morph_code']), axis=1)
    npv2021['npv'] = npv2021['npv'] - npv2021['debit']
    extra_capacity_21 = npv2021.apply(
        lambda x: extra_offload21(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).values.tolist()
    temp_21 = np.hstack(extra_capacity_21).T
    npv2021 = pd.concat([npv2021, temp_21], axis=1)

if len(npv2022) > 0:
    npv2022['npv'] = npv2022.apply(
        lambda x: npv22(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).astype(int)
    npv2022['split_yr'] = npv2022.apply(
        lambda x: split22(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1)
    npv2022['debit'] = npv2022.apply(
        lambda x: capex_22(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1) + \
                       npv2022.apply(lambda x: opex_22(x['gb_offload'], x['max_cap'],
                                                       x['morph_code']), axis=1)
    npv2022['npv'] = npv2022['npv'] - npv2022['debit']
    extra_capacity_22 = npv2022.apply(
        lambda x: extra_offload22(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).values.tolist()
    temp_22 = np.hstack(extra_capacity_22).T
    np_temp_22 = np.zeros([len(npv2022), 1])
    temp_22 = pd.DataFrame(np.concatenate((np_temp_22, temp_22), axis=1))
    npv2022 = pd.concat([npv2022, temp_22], axis=1)

if len(npv2023) > 0:
    npv2023['npv'] = npv2023.apply(
        lambda x: npv23(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).astype(int)
    npv2023['split_yr'] = npv2023.apply(
        lambda x: split23(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1)
    npv2023['debit'] = npv2023.apply(
        lambda x: capex_23(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1) + \
                       npv2023.apply(lambda x: opex_23(x['gb_offload'], x['max_cap'],
                                                       x['morph_code']), axis=1)
    npv2023['npv'] = npv2023['npv'] - npv2023['debit']
    extra_capacity_23 = npv2023.apply(
        lambda x: extra_offload23(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).values.tolist()
    temp_23 = np.hstack(extra_capacity_23).T
    np_temp_23 = np.zeros([len(npv2023), 2])
    temp_23 = pd.DataFrame(np.concatenate((np_temp_23, temp_23), axis=1))
    npv2023 = pd.concat([npv2023, temp_23], axis=1)

if len(npv2024) > 0:
    npv2024['npv'] = npv2024.apply(
        lambda x: npv24(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).astype(int)
    npv2024['split_yr'] = npv2024.apply(
        lambda x: split24(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1)
    npv2024['debit'] = npv2024.apply(
        lambda x: capex_24(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1) + \
                       npv2024.apply(lambda x: opex_24(x['gb_offload'], x['max_cap'],
                                                       x['morph_code']), axis=1)
    npv2024['npv'] = npv2024['npv'] - npv2024['debit']
    extra_capacity_24 = npv2024.apply(
        lambda x: extra_offload24(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).values.tolist()
    temp_24 = np.hstack(extra_capacity_24).T
    np_temp_24 = np.zeros([len(npv2024), 3])
    temp_24 = pd.DataFrame(np.concatenate((np_temp_24, temp_24), axis=1))
    npv2024 = pd.concat([npv2024, temp_24], axis=1)

if len(npv2025) > 0:
    npv2025['npv'] = npv2025.apply(
        lambda x: npv25(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).astype(int)
    npv2025['split_yr'] = npv2025.apply(
        lambda x: split25(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1)
    npv2025['debit'] = npv2025.apply(
        lambda x: capex_25(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1) + \
                       npv2025.apply(lambda x: opex_25(x['gb_offload'], x['max_cap'],
                                                       x['morph_code']), axis=1)
    npv2025['npv'] = npv2025['npv'] - npv2025['debit']
    extra_capacity_25 = npv2025.apply(
        lambda x: extra_offload25(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).values.tolist()
    temp_25 = np.hstack(extra_capacity_25).T
    np_temp_25 = np.zeros([len(npv2025), 4])
    temp_25 = pd.DataFrame(np.concatenate((np_temp_25, temp_25), axis=1))
    npv2025 = pd.concat([npv2025, temp_25], axis=1)

if len(npv2026) > 0:
    npv2026['npv'] = npv2026.apply(
        lambda x: npv26(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).astype(int)
    npv2026['split_yr'] = npv2026.apply(
        lambda x: split26(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1)
    npv2026['debit'] = npv2026.apply(
        lambda x: capex_26(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1) + \
                       npv2026.apply(lambda x: opex_26(x['gb_offload'], x['max_cap'],
                                                       x['morph_code']), axis=1)
    npv2026['npv'] = npv2026['npv'] - npv2026['debit']
    extra_capacity_26 = npv2026.apply(
        lambda x: extra_offload26(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).values.tolist()
    temp_26 = np.hstack(extra_capacity_26).T
    np_temp_26 = np.zeros([len(npv2026), 5])
    temp_26 = pd.DataFrame(np.concatenate((np_temp_26, temp_26), axis=1))
    npv2026 = pd.concat([npv2026, temp_26], axis=1)

if len(npv2027) > 0:
    npv2027['npv'] = npv2027.apply(
        lambda x: npv27(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).astype(int)
    npv2027['split_yr'] = npv2027.apply(
        lambda x: split27(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1)
    npv2027['debit'] = npv2027.apply(
        lambda x: capex_27(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1) + \
                       npv2027.apply(lambda x: opex_27(x['gb_offload'], x['max_cap'],
                                                       x['morph_code']), axis=1)
    npv2027['npv'] = npv2027['npv'] - npv2027['debit']
    extra_capacity_27 = npv2027.apply(
        lambda x: extra_offload27(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).values.tolist()
    temp_27 = np.hstack(extra_capacity_27).T
    np_temp_27 = np.zeros([len(npv2027), 6])
    temp_27 = pd.DataFrame(np.concatenate((np_temp_27, temp_27), axis=1))
    npv2027 = pd.concat([npv2027, temp_27], axis=1)

if len(npv2028) > 0:
    npv2028['npv'] = npv2028.apply(
        lambda x: npv28(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).astype(int)
    npv2028['split_yr'] = npv2028.apply(
        lambda x: split28(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1)
    npv2028['debit'] = npv2028.apply(
        lambda x: capex_28(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1) + \
                       npv2028.apply(lambda x: opex_28(x['gb_offload'], x['max_cap'],
                                                       x['morph_code']), axis=1)
    npv2028['npv'] = npv2028['npv'] - npv2028['debit']
    extra_capacity_28 = npv2028.apply(
        lambda x: extra_offload28(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).values.tolist()
    temp_28 = np.hstack(extra_capacity_28).T
    np_temp_28 = np.zeros([len(npv2028), 7])
    temp_28 = pd.DataFrame(np.concatenate((np_temp_28, temp_28), axis=1))
    npv2028 = pd.concat([npv2028, temp_28], axis=1)

if len(npv2029) > 0:
    npv2029['npv'] = npv2029.apply(
        lambda x: npv29(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).astype(int)
    npv2029['split_yr'] = npv2029.apply(
        lambda x: split29(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1)
    npv2029['debit'] = npv2029.apply(
        lambda x: capex_29(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1) + \
                       npv2029.apply(lambda x: opex_29(x['gb_offload'], x['max_cap'],
                                                       x['morph_code']), axis=1)
    npv2029['npv'] = npv2029['npv'] - npv2029['debit']
    extra_capacity_29 = npv2029.apply(
        lambda x: extra_offload29(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).values.tolist()
    temp_29 = np.hstack(extra_capacity_29).T
    np_temp_29 = np.zeros([len(npv2029), 8])
    temp_29 = pd.DataFrame(np.concatenate((np_temp_29, temp_29), axis=1))
    npv2029 = pd.concat([npv2029, temp_29], axis=1)

if len(npv2030) > 0:
    npv2030['npv'] = npv2030.apply(
        lambda x: npv30(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).astype(int)
    npv2030['split_yr'] = npv2030.apply(
        lambda x: split30(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1)
    npv2030['debit'] = npv2030.apply(
        lambda x: capex_30(x['gb_offload'], x['max_cap'], x['morph_code']), axis=1) + \
                       npv2030.apply(lambda x: opex_30(x['gb_offload'], x['max_cap'],
                                                       x['morph_code']), axis=1)
    npv2030['npv'] = npv2030['npv'] - npv2030['debit']
    extra_capacity_30 = npv2030.apply(
        lambda x: extra_offload30(x['gb_offload'], x['morph_code'], x['max_cap']),
        axis=1).values.tolist()
    temp_30 = np.hstack(extra_capacity_30).T
    np_temp_30 = np.zeros([len(npv2030), 9])
    temp_30 = pd.DataFrame(np.concatenate((np_temp_30, temp_30), axis=1))
    npv2030 = pd.concat([npv2030, temp_30], axis=1)

final = pd.concat(
    [npv2024, npv2025, npv2026, npv2027, npv2028, npv2029, npv2030, npv2021, npv2022,
     npv2023], sort=True, ignore_index=True)
# final = final.rename(columns={0: 'year_1_excess', 1: 'year_2_excess', 2: 'year_3_excess', 3: 'year_4_excess',
#                               4: 'year_5_excess', 5: 'year_6_excess', 6: 'year_7_excess', 7: 'year_8_excess',
#                               8: 'year_9_excess', 9: 'year_10_excess'})

# final = final[['fict_site', 'npv', 'fict_site_count', 'opt_build_year', 'gb_offload', 'max_cap', 'morph_code', 'debit',
#                'year_1_excess', 'year_2_excess', 'year_3_excess', 'year_4_excess', 'year_5_excess', 'year_6_excess',
#                'year_7_excess', 'year_8_excess', 'year_9_excess', 'year_10_excess']]

# final = final[['fict_site', 'npv', 'fict_site_count', 'opt_build_year', 'gb_offload', 'max_cap', 'morph_code', 'debit']]

final = final.sort_values(by=['fict_site'], ascending=True)
final = final.loc[final['npv'] > 0]

temp_final_npv = final[['npv']]
quant05 = temp_final_npv.quantile(.05).iloc[0]
quant10 = temp_final_npv.quantile(.1).iloc[0]
quant15 = temp_final_npv.quantile(.15).iloc[0]
quant20 = temp_final_npv.quantile(.2).iloc[0]
quant25 = temp_final_npv.quantile(.25).iloc[0]
quant30 = temp_final_npv.quantile(.3).iloc[0]
quant35 = temp_final_npv.quantile(.35).iloc[0]

print('final', end=' ')
print(final['npv'].sum())
print(len(final))

final_05 = final.loc[final['npv'] > quant05]
print('quant05', end=' ')
print(final_05['npv'].sum())
print(len(final_05))
print(quant05)

final_10 = final.loc[final['npv'] > quant10]
print('quant10', end=' ')
print(final_10['npv'].sum())
print(len(final_10))
print(quant10)

final_15 = final.loc[final['npv'] > quant15]
print('quant15', end=' ')
print(final_15['npv'].sum())
print(len(final_15))
print(quant15)

final_20 = final.loc[final['npv'] > quant20]
print('quant20', end=' ')
print(final_20['npv'].sum())
print(len(final_20))
print(quant20)

final_25 = final.loc[final['npv'] > quant25]
print('quant25', end=' ')
print(final_25['npv'].sum())
print(len(final_25))
print(quant25)

final_30 = final.loc[final['npv'] > quant30]
print('quant30', end=' ')
print(final_30['npv'].sum())
print(len(final_30))
print(quant30)

final_35 = final.loc[final['npv'] > quant35]
print('quant35', end=' ')
print(final_35['npv'].sum())
print(len(final_35))
print(quant35)

final['type'] = final.apply(lambda x: reverse_morph_type(x['morph_code']), axis=1)
final['morph'] = final.apply(lambda x: reverse_morph_morph(x['morph_code']), axis=1)
final['type_morph'] = final.apply(lambda x: reverse_morph_type_morph(x['morph_code']),
                                  axis=1)

sinr_test = sinr_test[['GridName']]
init_sites = pd.merge(sinr_test, init_sites, on='GridName')
final = pd.merge(final, init_sites, on='fict_site')

final['rx_signal_strength_db'], final['sinr'], final['rx_signal_strength_mw'] = \
    rx_calc(final['path_loss_umi_db'])
sinr_bins_unique = final.drop_duplicates(subset='GridName', keep=False).copy()
sinr_bins_dups = final[final.duplicated(['GridName'], keep=False)].copy()
sinr_bins_dups['sum_rx_signal'] = sinr_bins_dups.groupby('GridName')[
    'rx_signal_strength_mw'].transform(sum)
final = sinr_bins_unique.append(sinr_bins_dups)
sinr_bins_a = final[final['sum_rx_signal'].notnull()].copy().reset_index(drop=True)

sinr_bins_a['sinr_new'] = sinr_new2(sinr_bins_a['sum_rx_signal'],
                                    sinr_bins_a['rx_signal_strength_db'])

sinr_bins_b = final[final['sum_rx_signal'].isnull()].copy()
sinr_bins_b['sinr_new'] = sinr_bins_b['sum_rx_signal'].where(
    sinr_bins_b['sum_rx_signal'].notnull(),
    sinr_bins_b['sinr'], axis=0)
final = sinr_bins_a.append(sinr_bins_b)
final_bins = final[['GridName', 'sinr_new']]
final_bins_sinr = final_bins.groupby('GridName')['sinr_new'].mean().reset_index()
final_bins_sinr['sinr'] = round(final_bins_sinr['sinr_new'])
print('final_bins_sinr avg_sinr', end='  ')
print(round(final_bins_sinr['sinr'].mean()))
final_bins_sinr.to_csv('bins.csv')

print(timeit.default_timer() - init_time_2)

# end of program
