USE Walmart;

SELECT * FROM Region;

SELECT * FROM DMZ;

SELECT * FROM Demand
LIMIT 2000;

SELECT 
    table_schema 'Database Name',
    SUM(data_length + index_length) 'Size in Bytes',
    ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) 'Size in MiB'
FROM information_schema.tables 
GROUP BY table_schema;
