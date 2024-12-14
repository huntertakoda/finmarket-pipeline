package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
)

type TimeSeries struct {
	Open   string `json:"1. open"`
	High   string `json:"2. high"`
	Low    string `json:"3. low"`
	Close  string `json:"4. close"`
	Volume string `json:"5. volume"`
}

type Response struct {
	TimeSeries map[string]TimeSeries `json:"Time Series (5min)"`
}

func main() {
	apiKey := ""
	baseURL := "https://www.alphavantage.co/query"
	function := "TIME_SERIES_INTRADAY"
	symbol := "AAPL"
	interval := "5min"
	outputSize := "full" // fetch full dataset

	// construct the full url
	url := fmt.Sprintf("%s?function=%s&symbol=%s&interval=%s&outputsize=%s&apikey=%s",
		baseURL, function, symbol, interval, outputSize, apiKey)

	// send the http get request
	resp, err := http.Get(url)
	if err != nil {
		fmt.Printf("error making request: %v\n", err)
		return
	}
	defer resp.Body.Close()

	// read the response body
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("error reading response: %v\n", err)
		return
	}

	// parse the json response
	var data Response
	if err := json.Unmarshal(body, &data); err != nil {
		fmt.Printf("error parsing json: %v\n", err)
		return
	}

	// create a csv file
	file, err := os.Create("full_financial_data.csv")
	if err != nil {
		fmt.Printf("error creating file: %v\n", err)
		return
	}
	defer file.Close()

	// write data to csv
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// write header
	writer.Write([]string{"Timestamp", "Open", "High", "Low", "Close", "Volume"})

	// write rows
	for timestamp, ts := range data.TimeSeries {
		row := []string{timestamp, ts.Open, ts.High, ts.Low, ts.Close, ts.Volume}
		writer.Write(row)
	}

	fmt.Println("data saved to full_financial_data.csv successfully.")
}
