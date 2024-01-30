import dayjs from 'dayjs';
import { useEffect, useState } from 'react';
import { Button, Col, Container, Row, Table } from 'react-bootstrap';
import Chart from 'react-apexcharts';
import { Line } from 'react-chartjs-2';
import useAxios from './utils/useAxios';
import './index.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import {
    Chart as ChartJS,
    ArcElement,
    LineElement,
    BarElement,
    PointElement,
    BarController,
    BubbleController,
    DoughnutController,
    LineController,
    PieController,
    PolarAreaController,
    RadarController,
    ScatterController,
    CategoryScale,
    LinearScale,
    LogarithmicScale,
    RadialLinearScale,
    TimeScale,
    TimeSeriesScale,
    Decimation,
    Filler,
    Legend,
    Title,
    Tooltip,
} from 'chart.js';
import 'chartjs-adapter-dayjs-4/dist/chartjs-adapter-dayjs-4.esm';
import axios from 'axios';

ChartJS.register(
    ArcElement,
    LineElement,
    BarElement,
    PointElement,
    BarController,
    BubbleController,
    DoughnutController,
    LineController,
    PieController,
    PolarAreaController,
    RadarController,
    ScatterController,
    CategoryScale,
    LinearScale,
    LogarithmicScale,
    RadialLinearScale,
    TimeScale,
    TimeSeriesScale,
    Decimation,
    Filler,
    Legend,
    Title,
    Tooltip
);

function App() {
    const api = useAxios();
    const [data, setData] = useState([]);
    const [OHLC, setOHLC] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            let resp = await api.get('trade');
            console.log(resp.data);
            setData(
                resp.data.map((t) => {
                    t.c_profit = t.close_date && t.close_rate * t.amount - 1000;
                    return t;
                })
            );
        };
        const fetchOHLCData = async () => {
            let resp = await axios.get('https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1m&limit=1000');
            console.log(resp.data);
            setOHLC(resp.data.map((d) => d.slice(0, 5).map((i) => parseFloat(i))));
        };
        fetchData();
        fetchOHLCData();
    }, []);

    const chartData = {
        datasets: [
            {
                label: 'Cumulative Profit',
                data: data.map((t) => {
                    return { x: dayjs(t.open_date).format(), y: t.c_profit };
                }),
            },
        ],
    };

    const chartOptions = {
        scales: {
            x: {
                type: 'time',
            },
        },
    };

    return (
        <Container>
            <Row>
                <Col></Col>
                <Col xs="auto">
                    <h3>ETH OHLC</h3>
                </Col>
                <Col></Col>
            </Row>
            <Row>
                <Col>
                    <Chart
                        options={{
                            chart: {
                                type: 'candlestick',
                            },
                            title: {
                                text: 'ETH Candlestick Chart',
                            },
                            xaxis: {
                                type: 'datetime',
                            },
                            yaxis: {
                                tooltip: {
                                    enabled: true,
                                },
                            },
                        }}
                        series={[{ data: OHLC }]}
                        type="candlestick"
                    />
                    ;
                </Col>
            </Row>
            <Row>
                <Col></Col>
                <Col xs="auto">
                    <h3>Trades</h3>
                </Col>
                <Col></Col>
            </Row>
            <Row>
                <Col>
                    <Table>
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Amount</th>
                                <th>Open rate</th>
                                <th>Close rate</th>
                                <th>Profit</th>
                                <th>Open date</th>
                                <th>Close date</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.map((t) => (
                                <tr>
                                    <td>{t.id}</td>
                                    <td>{t.amount}</td>
                                    <td>{t.open_rate}</td>
                                    <td>{t.close_rate}</td>
                                    <td>
                                        {t.close_date ? (
                                            `${Math.round((t.close_rate / t.open_rate - 1) * 100000) / 1000}%`
                                        ) : (
                                            <Button onClick={() => api.post(`close/${t.id}`)}>Close</Button>
                                        )}
                                    </td>
                                    <td>{t.open_date}</td>
                                    <td>{t.close_date}</td>
                                </tr>
                            ))}
                        </tbody>
                    </Table>
                </Col>
            </Row>
            <Row>
                <Col></Col>
                <Col xs="auto">
                    <h3>Cumulative Profit</h3>
                </Col>
                <Col></Col>
            </Row>
            <Row>
                <Col>
                    <Line data={chartData} options={chartOptions} />
                </Col>
            </Row>
        </Container>
    );
}

export default App;
