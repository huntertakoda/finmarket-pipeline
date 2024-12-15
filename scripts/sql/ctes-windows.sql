# 7-day moving average of close prices

with moving_avg as (  
    select  
        timestamp,  
        close,  
        avg(close) over (order by timestamp rows between 6 preceding and current row) as moving_avg_close  
    from full_financial_data  
)  
select *  
from moving_avg  
order by timestamp
;  

# rank records by price volatility

with ranked_volatility as (  
    select  
        timestamp,  
        high,  
        low,  
        (high - low) as volatility,  
        rank() over (order by (high - low) desc) as rank_volatility  
    from full_financial_data  
)  
select *  
from ranked_volatility  
where rank_volatility <= 10
;  

# cumulative volume per day

with daily_volume as (  
    select  
        date(timestamp) as trade_date,  
        sum(volume) as total_volume  
    from full_financial_data  
    group by trade_date  
)  
select  
    trade_date,  
    total_volume,  
    sum(total_volume) over (order by trade_date) as cumulative_volume  
from daily_volume  
order by trade_date
;  

# calculate price change and rank it

with price_change as (  
    select  
        timestamp,  
        open,  
        close,  
        (close - open) as price_diff,  
        rank() over (order by (close - open) desc) as price_rank  
    from full_financial_data  
)  
select *  
from price_change  
where price_rank <= 5
;  

# rolling max and min for close price 

with rolling_extremes as (  
    select  
        timestamp,  
        close,  
        max(close) over (order by timestamp rows between 6 preceding and current row) as rolling_max_close,  
        min(close) over (order by timestamp rows between 6 preceding and current row) as rolling_min_close  
    from full_financial_data  
)  
select *  
from rolling_extremes  
order by timestamp
;  

# daily average close and volume summary

with daily_summary as (  
    select  
        date(timestamp) as trade_date,  
        avg(close) as avg_close_price,  
        avg(volume) as avg_volume  
    from full_financial_data  
    group by trade_date  
)  
select *  
from daily_summary  
order by trade_date
;  

# rank days by total trading volume

with daily_volume as (  
    select  
        date(timestamp) as trade_date,  
        sum(volume) as total_volume  
    from full_financial_data  
    group by trade_date  
)  
select  
    trade_date,  
    total_volume,  
    rank() over (order by total_volume desc) as volume_rank  
from daily_volume  
order by volume_rank
;  

# daily price volatility summary

with daily_volatility as (  
    select  
        date(timestamp) as trade_date,  
        max(high) - min(low) as daily_volatility  
    from full_financial_data  
    group by trade_date  
)  
select *  
from daily_volatility  
order by daily_volatility desc
;  

# rolling average of volume 

with rolling_volume as (  
    select  
        timestamp,  
        volume,  
        avg(volume) over (order by timestamp rows between 6 preceding and current row) as rolling_avg_volume  
    from full_financial_data  
)  
select *  
from rolling_volume  
order by timestamp
;  

# price change percentage per day

with daily_price_change as (  
    select  
        date(timestamp) as trade_date,  
        avg(open) as avg_open,  
        avg(close) as avg_close,  
        round(((avg(close) - avg(open)) / avg(open)) * 100, 2) as percent_change  
    from full_financial_data  
    group by trade_date  
)  
select *  
from daily_price_change  
order by percent_change desc
;  