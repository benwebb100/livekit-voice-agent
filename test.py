import requests
import json
from datetime import datetime, timedelta

# Hardcoded token (valid until expiry time)
auth_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6IjdHOEp0amo4bXF0WlF2djkifQ.eyJhdWQiOiJodHRwczovL2FwaS5kZWxpdmVyZWN0LmNvbSIsImV4cCI6MTc1MDgwMjgxNSwiaWF0IjoxNzUwNzE2NDE1LCJpc3MiOiJodHRwczovL2FwaS5kZWxpdmVyZWN0LmNvbSIsImF6cCI6IjFZcGFGZFNJWHJWT3BvQXIiLCJzdWIiOiIxWXBhRmRTSVhyVk9wb0FyQGNsaWVudHMiLCJzY29wZSI6ImdlbmVyaWNDaGFubmVsOmNlcnR1c2FpIn0.tMAU8-26ieMJWqf_RCZlj75OOMmO3V-LH4qt5xBD08cXMl_v8t8jBsCQCpHIInTdlh2wr3oyzNGU95NdEb-VYui5EJ7H5drZTJi0rBlZI2kRdp22puP0WfG9POmbkvpejW0HUEIOhPYseVJxMdh4gLw-cg0zZ2sJWobu4Fq7q7e9afEwjIO0G1Ou3wGPA4Iu12bNyuDmRZs4jLVbSSMze5OwaGAsKfVagPwDyxMfTWSnjo0JdZJOuM_vSDxP43-rEQ2Fmn7hubVPs1P0IA4BudFVYLzoa6CaHyRhqXZDdEl8EziCx_fzYFyFZjD3ly101Zl5Z_9sY8Fzfe-SAtCTpA"

# Headers with hardcoded token
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {auth_token}",
}

# Deliverect API endpoint
url = "https://api.deliverect.com/certusai/order/67cafd639889086e318f1065"

# Pickup time 1 hour from now (UTC ISO format)
pickup_time = (datetime.utcnow() + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

# Order payload
payload = {
    "customer": {
        "name": "Isaac Nichols",
        "companyName": "",
        "phoneNumber": "+18669044018",
        "email": ""
    },
    "payment": {
        "amount": 2180,
        "type": 1,
        "due": 2180
    },
    "taxes": [
        {
            "total": 180
        }
    ],
    "items": [
        {
            "plu": "P-BURG-CHK",
            "name": "Chicken Burger",
            "quantity": 1,
            "price": 800,
            "remark": "",
            "subItems": []
        },
        {
            "plu": "PIZZ-01",
            "name": "The Hawaiian",
            "quantity": 1,
            "price": 800,
            "remark": "",
            "subItems": []
        },
        {
            "plu": "DRNK-01",
            "name": "Coca Cola",
            "quantity": 1,
            "price": 400,
            "remark": "",
            "subItems": []
        }
    ],
    "channelLinkId": "67cafd639889086e318f1065",
    "channelOrderId": "4018_2310313231000",
    "channelOrderDisplayId": "4018_2310313231000",
    "orderType": 1,
    "deliveryIsAsap": False,
    "orderIsAlreadyPaid": False,
    "decimalDigits": 2,
    "note": "",
    "tip": 0,
    "pickupTime": pickup_time,
    "serviceCharge": 0,
    "serviceChargeTax": 0
}

# Send the POST request
response = requests.post(url, headers=headers, json=payload)

# Print result
print("Status Code:", response.status_code)
print("Response Body:", response.text)
