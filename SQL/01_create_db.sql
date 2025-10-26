CREATE DATABASE db_churn;
USE db_churn;

SELECT * FROM customer_data;

SELECT Gender, COUNT(Gender) AS count, COUNT(Gender)*100/(SELECT COUNT(*) FROM customer_data) AS percentage
FROM customer_data
GROUP BY Gender;


SELECT Contract, COUNT(Contract) AS count, COUNT(Contract)*100/(SELECT COUNT(*) FROM customer_data) AS percentage
FROM customer_data
GROUP BY Contract;


SELECT Customer_Status, COUNT(Customer_Status) AS count, COUNT(Customer_Status)*100/(SELECT COUNT(*) FROM customer_data) AS percentage
FROM customer_data
GROUP BY Customer_Status;


SELECT State, COUNT(State) AS count, COUNT(State)*100/(SELECT COUNT(*) FROM customer_data) AS percentage
FROM customer_data
GROUP BY State
ORDER BY percentage DESC;


SELECT Internet_Type, COUNT(Internet_Type) AS count, COUNT(Internet_Type)*100/(SELECT COUNT(*) FROM customer_data) AS percentage
FROM customer_data
GROUP BY Internet_Type;