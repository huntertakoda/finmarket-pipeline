# moving average of close prices (7-day window)

select  
    date(timestamp) as trade_date,  
    avg(close) over (order by date(timestamp) rows between 6 preceding and current row) as moving_avg_close  
from full_financial_data  
group by trade_date, close  
order by trade_date
;  

# daily highest volatility

select  
    date(timestamp) as trade_date,  
    max(high - low) as max_volatility  
from full_financial_data  
group by trade_date  
order by max_volatility desc
;  


# percentage change (close - open / open)

select  
    timestamp,  
    open,  
    close,  
    round(((close - open) / open) * 100, 2) as percent_change  
from full_financial_data  
order by percent_change desc
;  

# most active trading days (based on total volume)

select  
    date(timestamp) as trade_date,  
    sum(volume) as total_volume  
from full_financial_data  
group by trade_date  
order by total_volume desc  
limit 5
;  

# price spikes (close > 5% higher than open)

select *  
from full_financial_data  
where (close - open) / open > 0.05  
order by timestamp
;  

# cumulative total volume over time

with daily_volumes as (
    select 
        date(timestamp) as trade_date, 
        sum(volume) as total_volume
    from full_financial_data
    group by trade_date
)
select 
    trade_date, 
    sum(total_volume) over (order by trade_date) as cumulative_volume
from daily_volumes
order by trade_date
;

# days where volume > average volume

select  
    date(timestamp) as trade_date,  
    volume  
from full_financial_data  
where volume > (select avg(volume) from full_financial_data)  
order by volume desc
;  

# daily open-to-close price ratio

select  
    date(timestamp) as trade_date,  
    round(avg(close / open), 4) as avg_open_close_ratio  
from full_financial_data  
group by trade_date  
order by trade_date
;  

# rank records by price volatility

select  
    timestamp,  
    high,  
    low,  
    (high - low) as volatility,  
    rank() over (order by (high - low) desc) as volatility_rank  
from full_financial_data  
order by volatility_rank
;  

# price range summary per day

select  
    date(timestamp) as trade_date,  
    min(low) as min_low_price,  
    max(high) as max_high_price,  
    max(high) - min(low) as daily_range  
from full_financial_data  
group by trade_date  
order by daily_range desc
;  
