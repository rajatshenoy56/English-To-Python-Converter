import React, { Fragment, useState } from "react";
import { Button, FormControl, InputGroup } from "react-bootstrap";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import CodeEditor from "../../shared/CodeEditor/CodeEditor";

function TextInput() {
    const [text, setText] = useState("");
    const [disabled, setDisabled] = useState(true);
    const [codeText, setCodeText] = useState(null);

    function handleTextChange(e: {
        target: { value: React.SetStateAction<string> };
    }) {
        setText(e.target.value);
    }

    const showSuccessToast = () => {
        toast.info("Proper Sentence");
    };

    const showFailureToast = () => {
        toast.error("Improper Sentence");
    };

    function handleSubmit() {
        setDisabled(true);
        console.log(text);
        fetch("http://localhost:8000/input/text/", {
            body: JSON.stringify(text),
            cache: "no-cache",
            credentials: "same-origin",
            headers: {
                "content-type": "application/json",
            },
            method: "POST",
            mode: "cors",
            redirect: "follow",
            referrer: "no-referrer",
        }).then(async (response) => {
            const res = await response.json();
            if (String(res.lang) === "English") {
                if (localStorage.getItem("code") === null) {
                    localStorage.setItem("code", res.output);
                }
                //@ts-ignore
                else if (localStorage.getItem("code").charAt(localStorage.getItem("code").length-1) === ":"){
                    localStorage.setItem(
                        "code",
                        localStorage.getItem("code") + "\n" + "\t" + res.output
                    );
                }
                else{
                    localStorage.setItem(
                        "code",
                        localStorage.getItem("code") + "\n" +res.output
                    );
                }
                //@ts-ignore
                setCodeText(localStorage.getItem("code"));
                setDisabled(false);
                showSuccessToast();
            } else {
                showFailureToast();
            }
        });
    }
    return (
        <Fragment>
            <div style={{ padding: "50px" }}>
                <InputGroup className="mb-3">
                    <FormControl
                        placeholder="Code Text"
                        aria-label="Code Text"
                        aria-describedby="basic-addon2"
                        onChange={handleTextChange}
                    />
                    <InputGroup.Append>
                        <Button variant="success" onClick={handleSubmit}>
                            Submit
            </Button>
                    </InputGroup.Append>
                </InputGroup>
                <CodeEditor disabledVal={disabled} codeText={codeText} />
                <ToastContainer />
            </div>
        </Fragment>
    );
}

export default TextInput;
