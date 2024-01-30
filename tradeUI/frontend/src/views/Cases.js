import { useEffect, useState } from 'react';
import { Button, Col, Container, Row, Table } from 'react-bootstrap';
import { Link } from 'react-router-dom/cjs/react-router-dom.min';
import { minutesFromNow, formatDate } from '../utils/utils';
import './views.css';

import useAxios from '../utils/useAxios';

function Cases() {
    const api = useAxios();
    const [cases, setCases] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            let resp = await api.get('cases');
            setCases(resp.data);
        };
        fetchData();
    }, []);

    return (
        <>
            <Container>
                <Row className="m-1">
                    <Col>
                        <h2>Cases</h2>
                    </Col>
                </Row>
                <Row className="m-1">
                    <Col>
                        <Table>
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Subject</th>
                                    <th>Status</th>
                                    <th>Queue</th>
                                    <th>Priority</th>
                                    <th>Created</th>
                                    <th>Time left</th>
                                    <th>Owner</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {cases.map((c) => (
                                    <tr
                                        className={
                                            minutesFromNow(c.starts) < 0
                                                ? 'late'
                                                : minutesFromNow(c.starts) > 0 && minutesFromNow(c.starts) < 15
                                                ? 'upcoming'
                                                : ''
                                        }
                                        key={c.id}
                                    >
                                        <td>
                                            <Link to={`/cases/${c.id}`}>
                                                <Button> {c.id} </Button>
                                            </Link>
                                        </td>
                                        <td> {c.name} </td>
                                        <td> {c.status} </td>
                                        <td> {c.queue_fk.name} </td>
                                        <td> {c.priority} </td>
                                        <td> {formatDate(c.created)} </td>
                                        <td>
                                            {' '}
                                            {minutesFromNow(c.starts) >= 0
                                                ? minutesFromNow(c.starts)
                                                : minutesFromNow(c.starts) * -1}{' '}
                                            min {minutesFromNow(c.starts) < 0 ? 'ago' : ''}{' '}
                                        </td>
                                        <td>{c.owner}</td>
                                        <td>
                                            <Button> Take </Button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </Table>
                    </Col>
                </Row>
            </Container>
        </>
    );
}

export default Cases;
