# 下载CAMS的XCO2数据
import cdsapi

dataset = "cams-global-ghg-reanalysis-egg4-monthly"
request = {
    "variable": ["co2_column_mean_molar_fraction"],
    "year": ["2018"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "product_type": ["monthly_mean"],
    "data_format": "netcdf_zip",
    "area": [70, 55, 0, 137]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
