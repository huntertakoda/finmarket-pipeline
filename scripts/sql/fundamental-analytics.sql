select *
from full_financial_data
;

select count(*) as total_records  
from full_financial_data
;  

# 3996 records

select  
    avg(close) as avg_close_price,  
    min(close) as min_close_price,  
    max(close) as max_close_price  
from full_financial_data
;  

# avg, 236.87624351851898	min, 224.21	  max, 250.485

select  
    sum(volume) as total_volume,  
    avg(volume) as avg_volume  
from full_financial_data
;  

# tot. vol., 1066194802	avg. vol., 266815.5160

# daily summaries

select  
    date(timestamp) as trade_date,  
    count(*) as records,  
    avg(close) as avg_close,  
    sum(volume) as total_volume  
from full_financial_data  
group by trade_date  
order by trade_date asc
;  

# top 5 highest volume days

select  
    timestamp,  
    volume  
from full_financial_data  
order by volume desc  
limit 5
; 

# 2024-11-25 16:00:00	vol. : 87660160
# 2024-11-18 16:00:00	vol. : 24764651
# 2024-11-15 16:00:00	vol. : 22236107
# 2024-11-14 16:00:00	vol. : 18755749
# 2024-12-04 16:00:00	vol. : 18466300

# filter records > 1000000

select *  
from full_financial_data  
where volume > 1000000
;  

# close price volatility (high / low) per record

select  
    timestamp,  
    high,  
    low,  
    (high - low) as volatility  
from full_financial_data  
order by volatility desc
;  

# price changes (close / open)

select  
    timestamp,  
    open,  
    close,  
    (close - open) as price_change  
from full_financial_data  
order by price_change desc
;

# records with close price > open price

select *  
from full_financial_data  
where close > open
;  




