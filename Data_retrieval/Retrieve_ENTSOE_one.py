import requests

TOKEN = "25144c7e-835b-43a3-aef8-b653adff3248"
AREA = "10YNO-1--------2"

BASE_URL = "https://web-api.tp.entsoe.eu/api"

params = {
    "securityToken": TOKEN,
    "documentType": "A65",
    "processType": "A16",
    "outBiddingZone_Domain": AREA,
    "periodStart": "202501010000",
    "periodEnd": "202502010000"
}

response = requests.get(BASE_URL, params=params)

print(response.status_code)
print(response.url)
print(response.text[:3000])