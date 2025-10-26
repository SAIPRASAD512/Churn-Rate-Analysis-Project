SET SQL_SAFE_UPDATES = 0;

UPDATE customer_data
SET Value_Deal = NULL
WHERE TRIM(Value_Deal) = '';

UPDATE customer_data
SET Monthly_Charge = NULL
WHERE TRIM(Monthly_Charge) = '' OR Monthly_Charge < 0;

UPDATE customer_data
SET Internet_Type = NULL
WHERE TRIM(Internet_Type) = '';


UPDATE customer_data
SET 
    Customer_ID = NULLIF(TRIM(Customer_ID), ''),
    Gender = NULLIF(TRIM(Gender), ''),
    Age = NULLIF(TRIM(Age), ''),
    Married = NULLIF(TRIM(Married), ''),
    State = NULLIF(TRIM(State), ''),
    Number_of_Referrals = NULLIF(TRIM(Number_of_Referrals), ''),
    Tenure_in_Months = NULLIF(TRIM(Tenure_in_Months), ''),
    Value_Deal = NULLIF(TRIM(Value_Deal), ''),
    Phone_Service = NULLIF(TRIM(Phone_Service), ''),
    Multiple_Lines = NULLIF(TRIM(Multiple_Lines), ''),
    Internet_Service = NULLIF(TRIM(Internet_Service), ''),
    Internet_Type = NULLIF(TRIM(Internet_Type), ''),
    Online_Security = NULLIF(TRIM(Online_Security), ''),
    Online_Backup = NULLIF(TRIM(Online_Backup), ''),
    Device_Protection_Plan = NULLIF(TRIM(Device_Protection_Plan), ''),
    Premium_Support = NULLIF(TRIM(Premium_Support), ''),
    Streaming_TV = NULLIF(TRIM(Streaming_TV), ''),
    Streaming_Movies = NULLIF(TRIM(Streaming_Movies), ''),
    Streaming_Music = NULLIF(TRIM(Streaming_Music), ''),
    Unlimited_Data = NULLIF(TRIM(Unlimited_Data), ''),
    Contract = NULLIF(TRIM(Contract), ''),
    Paperless_Billing = NULLIF(TRIM(Paperless_Billing), ''),
    Payment_Method = NULLIF(TRIM(Payment_Method), ''),
    Monthly_Charge = NULLIF(TRIM(Monthly_Charge), ''),
    Total_Charges = NULLIF(TRIM(Total_Charges), ''),
    Total_Refunds = NULLIF(TRIM(Total_Refunds), ''),
    Total_Extra_Data_Charges = NULLIF(TRIM(Total_Extra_Data_Charges), ''),
    Total_Long_Distance_Charges = NULLIF(TRIM(Total_Long_Distance_Charges), ''),
    Total_Revenue = NULLIF(TRIM(Total_Revenue), ''),
    Customer_Status = NULLIF(TRIM(Customer_Status), ''),
    Churn_Category = NULLIF(TRIM(Churn_Category), ''),
    Churn_Reason = NULLIF(TRIM(Churn_Reason), '');

