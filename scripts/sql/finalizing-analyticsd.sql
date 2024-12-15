# identify outliers based on close price using z-score

with price_stats as (  
    select  
        avg(close) as mean_close,  
        stddev(close) as stddev_close  
    from full_financial_data  
),  
z_scores as (  
    select  
        timestamp,  
        close,  
        (close - price_stats.mean_close) / price_stats.stddev_close as z_score  
    from full_financial_data, price_stats  
)  
select *  
from z_scores  
where abs(z_score) > 3  
order by z_score desc
;  

# identifying gaps in timestamps

with gaps as (  
    select  
        timestamp,  
        lag(timestamp) over (order by timestamp) as previous_timestamp,  
        timediff(timestamp, lag(timestamp) over (order by timestamp)) as time_gap  
    from full_financial_data  
)  
select *  
from gaps  
where time_gap > '01:00:00'  
order by time_gap desc
;  

# cumulative daily volume percentage

with daily_volume as (  
    select  
        date(timestamp) as trade_date,  
        sum(volume) as total_volume  
    from full_financial_data  
    group by trade_date  
),  
cumulative as (  
    select  
        trade_date,  
        total_volume,  
        sum(total_volume) over (order by trade_date) as cumulative_volume,  
        sum(total_volume) over (order by trade_date) * 100 / sum(total_volume) over () as cumulative_percentage  
    from daily_volume  
)  
select *  
from cumulative  
order by trade_date
;  

# detecting days with sudden volume spikes

with volume_spikes as (  
    select  
        date(timestamp) as trade_date,  
        volume,  
        lag(volume) over (order by timestamp) as previous_volume,  
        (volume - lag(volume) over (order by timestamp)) as volume_difference  
    from full_financial_data  
)  
select *  
from volume_spikes  
where volume_difference > (previous_volume * 2)  
order by volume_difference desc
;  

# trend analysis: 3-day streak of rising close prices

with price_streaks as (  
    select  
        timestamp,  
        close,  
        lead(close, 1) over (order by timestamp) as next_close_1,  
        lead(close, 2) over (order by timestamp) as next_close_2  
    from full_financial_data  
)  
select *  
from price_streaks  
where close < next_close_1 and next_close_1 < next_close_2
;  

####

# test cases

# test for missing values

select *  
from full_financial_data  
where open is null  
   or close is null  
   or high is null  
   or low is null
  ;  

# test for duplicate records
 
select timestamp, open, close, count(*) as duplicate_count  
from full_financial_data  
group by timestamp, open, close  
having count(*) > 1
;  

# test for extreme values in close prices

select *  
from full_financial_data  
where close > 10000 or close < 0
;  

# test for zero or negative volumes

select *  
from full_financial_data  
where volume <= 0
;  

# test for invalid timestamp formats

select *  
from full_financial_data  
where timestamp not like '%-%-% %:%:%'
;  

# test for volatility outliers

select *  
from full_financial_data  
where (high - low) > 100
;

