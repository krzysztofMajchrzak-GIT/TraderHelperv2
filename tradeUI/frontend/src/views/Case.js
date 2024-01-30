import dayjs from 'dayjs';
import { useContext, useEffect, useRef, useState } from 'react';
import {
    Button,
    Col,
    Container,
    FloatingLabel,
    Form,
    OverlayTrigger,
    Row,
    Tooltip,
    Table,
    Modal,
} from 'react-bootstrap';
import { PencilSquare, PlusLg } from 'react-bootstrap-icons';
import { useParams } from 'react-router-dom';
import AuthContext from '../context/AuthContext';
import useAxios from '../utils/useAxios';
import AlertTable from '../components/alert/AlertTable';
import CaseData from '../components/case/CaseData';
import SocActions from '../components/elements/SocActions';
import Notes from '../components/elements/Notes';
import History from '../components/elements/History';
import TaxonomyComponent from '../components/elements/Taxonomy';
import ControlledEditor from '../components/Editor';
import Contacts from '../components/elements/Contacts';
import TI from '../components/elements/TI';

const COLOURS = {
    default: 'bg-secondary',
    open: 'bg-primary',
    closed: 'bg-success',
    skipped: 'bg-danger',
};

const PLACEHOLDERS = {
    EN: 'Unavailable in English',
    PL: 'Unavailable in Polish',
};

const VARIABLES = [
    [
        'platform_fk',
        (k, v, n) =>
            v
                ? [
                      Object.fromEntries([
                          ['text', `${k.slice(0, -3).toUpperCase()}${n + 1}: ${v.name}`],
                          ['value', `${k}${n}`],
                      ]),
                  ]
                : [],
    ],
    [
        'platform',
        (k, v, n) =>
            v
                ? [
                      Object.fromEntries([
                          ['text', `${k.toUpperCase()}${n + 1}: ${v.name}`],
                          ['value', `${k}${n}`],
                      ]),
                  ]
                : [],
    ],
    [
        'event_fk',
        (k, v, n) =>
            v
                ? [
                      Object.fromEntries([
                          ['text', `${k.slice(0, -3).toUpperCase()}${n + 1}: ${v.name}`],
                          ['value', `${k}${n}`],
                      ]),
                  ]
                : [],
    ],
    [
        'dst_ip',
        (k, v, n) =>
            v.map((a, i) =>
                Object.fromEntries([
                    ['text', `${k.toUpperCase()}${n + i + 1}: ${a.address}`],
                    ['value', `${k}${n + i}`],
                ])
            ),
    ],
    [
        'src_ip',
        (k, v, n) =>
            v.map((a, i) =>
                Object.fromEntries([
                    ['text', `${k.toUpperCase()}${n + i + 1}: ${a.address}`],
                    ['value', `${k}${n + i}`],
                ])
            ),
    ],
    [
        'files',
        // 'files', path
        // 'files', hash
        // 'software'
        (k, v, n) =>
            v.map((a, i) =>
                Object.fromEntries([
                    ['text', `${k.slice(0, -1).toUpperCase()}${n + i + 1}: ${a.filename_fk.name}`],
                    ['value', `${k.slice(0, -1)}${n + i}`],
                ])
            ),
    ],
    [
        'fqdns',
        (k, v, n) =>
            v.map((a, i) =>
                Object.fromEntries([
                    ['text', `${k.slice(0, -1).toUpperCase()}${n + i + 1}: ${a.address}`],
                    ['value', `${k.slice(0, -1)}${n + i}`],
                ])
            ),
    ],
    [
        'assets',
        (k, v, n) =>
            v.map((a, i) =>
                Object.fromEntries([
                    ['text', `${k.slice(0, -1).toUpperCase()}${n + i + 1}: ${a.name}`],
                    ['value', `${k.slice(0, -1)}${n + i}`],
                ])
            ),
    ],
    [
        'users',
        (k, v, n) =>
            v.map((a, i) =>
                Object.fromEntries([
                    ['text', `${k.slice(0, -1).toUpperCase()}${n + i + 1}: ${a.name}`],
                    ['value', `${k.slice(0, -1)}${n + i}`],
                ])
            ),
    ],
    [
        'system_actions',
        (k, v, n) =>
            v.map((a, i) =>
                Object.fromEntries([
                    ['text', `${k.slice(0, -1).toUpperCase()}${n + i + 1}: ${a.name}`],
                    ['value', `${k.slice(0, -1)}${n + i}`],
                ])
            ),
    ],
    [
        'indicators',
        (k, v, n) =>
            v.map((a, i) =>
                Object.fromEntries([
                    ['text', `${k.slice(0, -1).toUpperCase()}${n + i + 1}: ${a.name}`],
                    ['value', `${k.slice(0, -1)}${n + i}`],
                ])
            ),
    ],
    [
        'processes',
        (k, v, n) =>
            v.map((a, i) =>
                Object.fromEntries([
                    ['text', `${k.slice(0, -2).toUpperCase()}${n + i + 1}: ${a.name}`],
                    ['value', `${k.slice(0, -2)}${n + i}`],
                ])
            ),
    ],
];

function Case() {
    const api = useAxios();
    const { id } = useParams();
    const [caseData, setCaseData] = useState({});
    const [SOCActions, setSOCActions] = useState([]);
    const [taxonomies, setTaxonomies] = useState([]);
    const [notes, setNotes] = useState([]);
    const [variables, setVariables] = useState({});
    const [language, setLanguage] = useState('EN');

    useEffect(() => {
        const fetchData = async () => {
            let resp = await api.get(`${id}`);
            setCaseData(resp.data);
            setVariables(
                VARIABLES.reduce(
                    (r, [k, v]) =>
                        r.concat(resp.data.alert_set.reduce((as, a) => as.concat(v(k, a[k], as.length)), [])),
                    []
                )
            );
            setLanguage(resp.data.queue_fk.language);

            const respTaxo = await api.get(`queue/${resp.data.queue_fk.id}/taxonomy`);
            const taxoData = respTaxo.data;

            setTaxonomies(taxoData);

            const respNotes = await api.get(`case/${id}/notes/`);
            setNotes(respNotes.data);
            console.log(respNotes.data);
        };
        const fetchSOCActions = async () => {
            let resp = await api.get('socaction');
            setSOCActions(resp.data);
        };
        fetchSOCActions();
        fetchData();
    }, []);

    return (
        <>
            {JSON.stringify(variables)}
            <Container>
                <CaseData caseData={caseData}>
                    {SOCActions.length > 0 && caseData.soc_actions ? (
                        <SocActions selectedSOCActions={caseData.soc_actions} SOCActions={SOCActions} />
                    ) : (
                        <></>
                    )}
                </CaseData>
            </Container>
            <Row>
                <Col>
                    <AlertTable alert_set={caseData.alert_set} />
                </Col>
            </Row>
            <Container>
                <Row className="mx-5">
                    <Col>
                        <h4>Analysis</h4>
                    </Col>
                    <Col xs="auto">
                        <h5>Select language:</h5>
                    </Col>
                    <Col xs="auto">
                        <LanguageSwitch language={language} setLanguage={setLanguage} />
                    </Col>
                </Row>
                <Row>
                    <Col>
                        <Analysis language={language} variables={variables} caseID={id} />
                    </Col>
                </Row>
            </Container>
            <Container>
                <Row className="mx-5">
                    <Col>
                        <Notes notes={notes} />
                    </Col>
                </Row>
                <Row className="mx-5">
                    <Col>
                        <TI />
                    </Col>
                </Row>
                <Row className="mx-5">
                    <Col>
                        <TaxonomyComponent
                            taxonomies={taxonomies}
                            severity={caseData.severity}
                            name={caseData.classification}
                        />
                    </Col>
                </Row>
                <Row className="mx-5">
                    <Col>
                        <Contacts />
                    </Col>
                </Row>
                <Row className="mx-5">
                    <Col>
                        <History />
                    </Col>
                </Row>
            </Container>
        </>
    );
}

function Analysis({ language, variables, caseID }) {
    const api = useAxios();
    const [forest, setForest] = useState([]);
    const [questions, setQuestions] = useState([]);
    const [currentAnswer, setCurrentAnswer] = useState(null);
    const [currentQuestionID, setCurrentQuestionID] = useState(null);
    const [started, setStarted] = useState(dayjs());
    const [analysis, setAnalysis] = useState({});

    useEffect(() => {
        const fetchData = async () => {
            let resp = await api.get('question_tree');
            setForest(Object.fromEntries(resp.data.map((q) => [q.id, q])));
        };
        fetchData();
    }, []);

    useEffect(async () => {
        let resp = await api.get(`anl/${caseID}`);
        const previousAnalysis = Object.fromEntries(
            resp.data
                .sort((a, b) => a.order - b.order)
                .map((ca) => [
                    ca.question,
                    { answer: ca.answer, order: ca.order, started: ca.started, finished: ca.finished },
                ])
        );
        Object.values(forest).forEach((tree) => {
            parseTree(tree, 0, previousAnalysis);
        });
        setAnalysis(previousAnalysis);
        setCurrentQuestionID(Object.keys(forest)[0]);
        setCurrentAnswer({ id: 1, child_questions: Object.keys(forest) });
    }, [forest]);

    useEffect(() => {
        const currentAnswerIndex = questions[currentQuestionID]?.answers.findIndex((a) => a.status);
        if (currentAnswerIndex >= 0) setCurrentAnswer(questions[currentQuestionID].answers[currentAnswerIndex]);
        else if (!questions[currentQuestionID]?.depth)
            setCurrentAnswer({ id: 1, child_questions: Object.keys(forest) });
        else setCurrentAnswer(null);
        setStarted(dayjs());
    }, [currentQuestionID]);

    function parseTree(tree, depth, previous = {}, parent_id = null) {
        setQuestions((questions) => {
            return {
                ...questions,
                [tree.id]: {
                    id: tree.id,
                    content: tree.content,
                    content_pl: tree.content_pl,
                    description: tree.description,
                    link: tree.link,
                    answers: tree.child_answer_set.map((a) => {
                        return {
                            id: a.id,
                            content: a.content,
                            content_pl: a.content_pl,
                            child_questions: a.child_questions.map((q) => q.id),
                            status: previous[tree.id]?.answer === a.id,
                        };
                    }),
                    depth: depth,
                    order: Object.keys(questions).length,
                    parent_id: parent_id,
                    status: previous[tree.id]
                        ? 'closed'
                        : previous[parent_id]
                        ? 'default'
                        : depth
                        ? 'hidden'
                        : 'default',
                },
            };
        });
        tree.child_answer_set.forEach((a) => {
            a.child_questions.forEach((q) => {
                parseTree(q, depth + 1, previous, tree.id);
            });
        });
    }

    const currentQuestion = questions[currentQuestionID];
    return (
        <Container>
            <p>{JSON.stringify(analysis)}</p>
            <Row>
                <Col>
                    <Row>
                        <Col>
                            <QuestionTree
                                language={language}
                                questions={questions}
                                currentQuestionID={currentQuestionID}
                                setCurrentQuestionID={setCurrentQuestionID}
                            />
                        </Col>
                    </Row>
                </Col>
                <Col>
                    <Row>
                        <Col>
                            {currentQuestionID && (
                                <Question
                                    language={language}
                                    variables={variables}
                                    questions={questions}
                                    setQuestions={setQuestions}
                                    currentQuestion={currentQuestion}
                                    updateAnalysis={(newAnswerID, next, newAnswer = null) => {
                                        let answers = currentQuestion.answers;
                                        if (newAnswer) {
                                            answers = answers.concat([newAnswer]);
                                        } else {
                                            newAnswer = answers[answers.findIndex((a) => a.id === newAnswerID)];
                                        }
                                        answers.forEach((a) => {
                                            a.status = false;
                                        });
                                        newAnswer.status = true;
                                        const newQuestions = questions;
                                        Object.values(newQuestions).forEach((q) => {
                                            if (q.status === 'default' && q.parent_id == currentQuestionID)
                                                q.status = 'hidden';
                                        });
                                        newQuestions[currentQuestionID] = {
                                            ...currentQuestion,
                                            status: 'closed',
                                            answers: answers,
                                        };
                                        next.forEach((q) => {
                                            if (newQuestions[q].status === 'hidden') newQuestions[q].status = 'default';
                                        });
                                        setQuestions(newQuestions);
                                        setAnalysis({
                                            ...analysis,
                                            [currentQuestionID]: {
                                                answer: newAnswerID,
                                                order:
                                                    analysis[currentQuestionID]?.order ||
                                                    Object.keys(analysis).length + 1,
                                                started: started.format(),
                                                finished: dayjs().format(),
                                            },
                                        });
                                        setCurrentAnswer(newAnswer);
                                    }}
                                    setCurrentQuestionID={setCurrentQuestionID}
                                    currentAnswer={currentAnswer}
                                    updateTree={(answer, question) => {
                                        const answers = currentQuestion.answers;
                                        const answerIndex = answers.findIndex((a) => a.id === answer.id);
                                        if (answerIndex >= 0)
                                            answers[answerIndex] = { ...answers[answerIndex], ...answer };
                                        setQuestions({ ...questions, [question.id]: question });
                                        setCurrentQuestionID(question.id);
                                    }}
                                />
                            )}
                        </Col>
                    </Row>
                </Col>
            </Row>
            <Row>
                <Col></Col>
                <Col xs="auto">
                    <Button variant="dark" onClick={async () => await api.put(`${caseID}`, analysis)}>
                        Save
                    </Button>
                </Col>
                <Col></Col>
            </Row>
        </Container>
    );
}

function QuestionTree({ language, questions, currentQuestionID, setCurrentQuestionID }) {
    return (
        <Row>
            <Col>
                {Object.values(questions)
                    .sort((a, b) => a.order - b.order)
                    .map((q) => {
                        return (
                            q.status !== 'hidden' && (
                                <Row key={q.id} className="m-0">
                                    {q.depth ? <Col xs={q.depth}></Col> : <></>}
                                    <Col>
                                        <QuestionNode
                                            language={language}
                                            question={q}
                                            currentQuestionID={currentQuestionID}
                                            setCurrentQuestionID={setCurrentQuestionID}
                                        />
                                    </Col>
                                </Row>
                            )
                        );
                    })}
            </Col>
        </Row>
    );
}

function QuestionNode({ language, question, currentQuestionID, setCurrentQuestionID }) {
    return (
        <Node
            content={(language === 'EN' ? question.content : question.content_pl) || PLACEHOLDERS[language]}
            maxLines={1}
            className={
                'text-light border border-dark rounded m-0 ' +
                COLOURS[question.id == currentQuestionID ? 'open' : question.status]
            }
            onClick={() => setCurrentQuestionID(question.id)}
        />
    );
}

function Question({
    language,
    variables,
    questions,
    setQuestions,
    currentQuestion,
    updateAnalysis,
    setCurrentQuestionID,
    currentAnswer,
    updateTree,
}) {
    const api = useAxios();
    const [nextQuestionIDs, setNextQuestionIDs] = useState(null);
    const [showQuestionModal, setShowQuestionModal] = useState(false);
    const [showAnswerModal, setShowAnswerModal] = useState(false);
    const [editAnswer, setEditAnswer] = useState(null);

    const nextQuestions = Object.values(questions).filter((q) => nextQuestionIDs?.includes(q.id));
    if (!nextQuestions.length)
        nextQuestions.push(
            ...Object.values(questions).filter((q) => q.status === 'default' && q.id !== currentQuestion.id)
        );
    return (
        <Container>
            <Row>
                <Col>
                    <QuestionHeader
                        language={language}
                        question={currentQuestion}
                        setShowQuestionModal={setShowQuestionModal}
                    />
                </Col>
            </Row>
            <Row>
                <Col>
                    <Answers
                        language={language}
                        variables={variables}
                        answers={currentQuestion.answers}
                        setEditAnswer={setEditAnswer}
                        setShowAnswerModal={setShowAnswerModal}
                        updateAnalysis={(value, next) => {
                            setNextQuestionIDs(next);
                            updateAnalysis(value, next);
                        }}
                        onSave={async (answer) => {
                            let status = null;
                            await api
                                .post('answer/', {
                                    content: answer.content,
                                    content_pl: answer.content_pl,
                                    parent_question: currentQuestion.id,
                                })
                                .then((resp) => {
                                    console.log(resp);
                                    status = resp.status;
                                    setNextQuestionIDs([]);
                                    updateAnalysis(resp.data.id, [], resp.data);
                                })
                                .catch((error) => alert(error));
                            return status === 201;
                        }}
                    />
                </Col>
            </Row>
            <Row className="mb-5">
                <Col>
                    <NextQuestions
                        language={language}
                        variables={variables}
                        questions={nextQuestions}
                        setCurrentQuestionID={setCurrentQuestionID}
                        currentAnswer={currentAnswer}
                        onSave={async (questionData) => {
                            let status = null;
                            await api
                                .post('question/', {
                                    content: questionData.content,
                                    content_pl: questionData.content_pl,
                                    description: questionData.description,
                                    link: questionData.link,
                                })
                                .then(async (resp) => {
                                    const question = {
                                        ...resp.data,
                                        ...(currentAnswer.id > 1
                                            ? {
                                                  parent_id: currentQuestion.id,
                                                  order: currentQuestion.order + nextQuestionIDs.length + 0.01,
                                                  depth: currentQuestion.depth + 1,
                                              }
                                            : {
                                                  parent_id: null,
                                                  order: Object.keys(questions).length,
                                                  depth: 0,
                                              }),
                                        status: 'default',
                                        answers: [],
                                    };
                                    console.log(resp);
                                    resp = await api.patch(`answer/${currentAnswer.id}/`, {
                                        child_questions: currentAnswer.child_questions.concat([resp.data.id]),
                                    });
                                    console.log(resp);
                                    status = resp.status;
                                    updateTree(resp.data, question);
                                })
                                .catch((error) => alert(error));
                            return status === 201;
                        }}
                    />
                </Col>
            </Row>
            <NodeModal
                node={currentQuestion}
                updateNode={async (question) => {
                    let status = null;
                    await api
                        .patch(`question/${question.id}/`, {
                            content: question.content,
                            content_pl: question.content_pl,
                            description: question.description,
                            link: question.link,
                        })
                        .then((resp) => {
                            console.log(resp);
                            status = resp.status;
                            setQuestions({ ...questions, [question.id]: question });
                        })
                        .catch((error) => alert(error));
                    return status === 200;
                }}
                show={showQuestionModal}
                setShow={setShowQuestionModal}
                initLanguage={language}
                variables={variables}
            />
            {editAnswer && (
                <NodeModal
                    node={editAnswer}
                    updateNode={async (answer) => {
                        let status = null;
                        await api
                            .patch(`answer/${answer.id}/`, {
                                content: answer.content,
                                content_pl: answer.content_pl,
                            })
                            .then((resp) => {
                                console.log(resp);
                                status = resp.status;
                                setQuestions((questions) => {
                                    const editedAnswerID = currentQuestion.answers.findIndex((a) => a.id === answer.id);
                                    questions[currentQuestion.id].answers[editedAnswerID] = {
                                        ...questions[currentQuestion.id].answers[editedAnswerID],
                                        ...answer,
                                    };
                                    return questions;
                                });
                            })
                            .catch((error) => alert(error));
                        return status === 200;
                    }}
                    show={showAnswerModal}
                    setShow={setShowAnswerModal}
                    initLanguage={language}
                    variables={variables}
                    contentOnly
                />
            )}
        </Container>
    );
}

function QuestionHeader({ language, question, setShowQuestionModal }) {
    return (
        <Container>
            <Row>
                <Col>
                    <h2
                        dangerouslySetInnerHTML={{
                            __html:
                                (language === 'EN' ? question.content : question.content_pl) || PLACEHOLDERS[language],
                        }}
                    ></h2>
                </Col>
                <Col xs="auto">
                    <Button variant="outline-dark" onClick={() => setShowQuestionModal(true)}>
                        <PencilSquare size={24} />
                    </Button>
                </Col>
            </Row>
            <Row>
                <Col>{question.description}</Col>
            </Row>
        </Container>
    );
}

function NodeModal({ node, updateNode, show, setShow, initLanguage, variables, contentOnly }) {
    const [editor, setEditor] = useState(null);
    const [language, setLanguage] = useState(initLanguage);
    const [content, setContent] = useState(node.content);
    const [content_pl, setContent_pl] = useState(node.content_pl);
    const [description, setDescription] = useState(node.description);
    const [link, setLink] = useState(node.link);

    useEffect(() => setLanguage(initLanguage), [initLanguage]);

    useEffect(() => setEditorLanguage(), [language, variables]);

    useEffect(() => ResetModal(), [node.id]);

    function setEditorLanguage() {
        if (language === 'EN')
            setEditor(<ControlledEditor value={content} setValue={setContent} variables={variables} />);
        else setEditor(<ControlledEditor value={content_pl} setValue={setContent_pl} variables={variables} />);
    }

    function ResetModal() {
        setContent(node.content);
        setContent_pl(node.content_pl);
        setDescription(node.description);
        setLink(node.link);
        setLanguage(initLanguage);
        setEditorLanguage();
    }

    return (
        <Modal show={show} onHide={() => setShow(false)}>
            <Modal.Header closeButton>
                <Modal.Title>Edit question</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Form onSubmit={(e) => e.preventDefault()}>
                    <Form.Group className="mb-3">
                        <Row className="mx-5">
                            <Col xs="auto">
                                <Form.Label>Select language:</Form.Label>
                            </Col>
                            <Col xs="auto">
                                <LanguageSwitch language={language} setLanguage={setLanguage} />
                            </Col>
                        </Row>
                    </Form.Group>
                    <Form.Group className="mb-3">
                        <Form.Label>Content ({language === 'EN' ? 'English' : 'Polish'})</Form.Label>
                        {editor}
                    </Form.Group>
                    {!contentOnly && (
                        <>
                            <Form.Group className="mb-3">
                                <Form.Label>Description</Form.Label>
                                <Form.Control
                                    as="textarea"
                                    type="text"
                                    value={description}
                                    onChange={(e) => setDescription(e.target.value)}
                                />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Link</Form.Label>
                                <Form.Control type="text" value={link} onChange={(e) => setLink(e.target.value)} />
                            </Form.Group>
                        </>
                    )}
                </Form>
            </Modal.Body>
            <Modal.Footer>
                <Button
                    variant="secondary"
                    onClick={() => {
                        ResetModal();
                        setShow(false);
                    }}
                >
                    Discard
                </Button>
                <Button
                    variant="dark"
                    onClick={async () =>
                        setShow(!(await updateNode({ ...node, content, content_pl, description, link })))
                    }
                >
                    Save Changes
                </Button>
            </Modal.Footer>
        </Modal>
    );
}

function LanguageSwitch({ language, setLanguage }) {
    return (
        <Row>
            <Col xs="auto">
                <div
                    style={{
                        userSelect: 'none',
                    }}
                    onClick={() => setLanguage('EN')}
                >
                    English
                </div>
            </Col>
            <Col xs="auto" className="align-self-center">
                <Form.Check
                    type="switch"
                    checked={language === 'PL'}
                    onChange={() => setLanguage(language === 'EN' ? 'PL' : 'EN')}
                />
            </Col>
            <Col xs="auto">
                <div
                    style={{
                        userSelect: 'none',
                    }}
                    onClick={() => setLanguage('PL')}
                >
                    Polish
                </div>
            </Col>
        </Row>
    );
}

function Answers({ language, variables, answers, setEditAnswer, setShowAnswerModal, updateAnalysis, onSave }) {
    return (
        <Container>
            <Row>
                <Col>
                    <h3>Answer</h3>
                </Col>
            </Row>
            {answers.map((a) => (
                <Row key={a.id} className="g-0">
                    <Col>
                        <Node
                            content={(language === 'EN' ? a.content : a.content_pl) || PLACEHOLDERS[language]}
                            maxLines={5}
                            status={a.status}
                            onClick={() => updateAnalysis(a.id, a.child_questions)}
                        />
                    </Col>
                    <Col xs="auto">
                        <Button
                            variant="outline-dark"
                            className="p-0 d-flex align-items-center"
                            onClick={() => {
                                setEditAnswer(a);
                                setShowAnswerModal(true);
                            }}
                        >
                            <PencilSquare size={24} />
                        </Button>
                    </Col>
                </Row>
            ))}
            <Row>
                <Col>
                    <NewNode onSave={onSave} language={language} variables={variables} contentOnly />
                </Col>
            </Row>
        </Container>
    );
}

function NextQuestions({ language, variables, questions, setCurrentQuestionID, currentAnswer, onSave }) {
    return (
        <Container>
            <Row>
                <Col>
                    <h3>Recommended next questions</h3>
                </Col>
            </Row>
            {questions.map((q) => (
                <Row key={q.id}>
                    <Col>
                        <Node
                            content={(language === 'EN' ? q.content : q.content_pl) || PLACEHOLDERS[language]}
                            maxLines={5}
                            onClick={() => setCurrentQuestionID(q.id)}
                        />
                    </Col>
                </Row>
            ))}
            {currentAnswer && (
                <Row>
                    <Col>
                        <NewNode onSave={onSave} language={language} variables={variables} />
                    </Col>
                </Row>
            )}
        </Container>
    );
}

function Node({ content, className, status, maxLines, onClick }) {
    const [show, setShow] = useState(false);
    const ref = useRef(null);

    useEffect(() => {
        setShow(ref.current?.clientHeight < ref.current?.scrollHeight);
    }, [ref]);

    return (
        <ContentTooltip show={show}>
            <p
                className={className || 'border border-dark rounded m-0' + (status ? ' bg-success' : '')}
                style={{
                    overflow: 'hidden',
                    display: '-webkit-box',
                    WebkitBoxOrient: 'vertical',
                    WebkitLineClamp: maxLines,
                }}
                onClick={onClick}
                ref={ref}
                dangerouslySetInnerHTML={{
                    __html: content,
                }}
            ></p>
        </ContentTooltip>
    );
}

function NewNode({ onSave, language, variables, contentOnly }) {
    const [showModal, setShowModal] = useState(false);

    return (
        <div className="d-grid gy-0">
            <Button
                variant="outline-dark"
                className="d-flex align-items-center justify-content-center"
                onClick={() => setShowModal(true)}
            >
                <PlusLg size={12} />
            </Button>
            <NodeModal
                node={{
                    content: '',
                    content_pl: '',
                    description: '',
                    link: '',
                }}
                updateNode={(newNode) => onSave(newNode)}
                show={showModal}
                setShow={setShowModal}
                initLanguage={language}
                variables={variables}
                contentOnly={contentOnly}
            />
        </div>
    );
}

function ContentTooltip({ children, show }) {
    return (
        <OverlayTrigger
            trigger={show ? ['hover', 'focus'] : []}
            overlay={(props) => {
                return (
                    <Tooltip {...props}>
                        <p dangerouslySetInnerHTML={children.props.dangerouslySetInnerHTML}></p>
                    </Tooltip>
                );
            }}
        >
            {children}
        </OverlayTrigger>
    );
}

export default Case;
