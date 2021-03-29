import React, { useEffect } from "react";
import Editor from "react-simple-code-editor";
import { highlight, languages } from "prismjs/components/prism-core";
import "prismjs/components/prism-clike";
import "prismjs/components/prism-python";
import "prismjs/themes/prism-okaidia.css";
import { Button } from "react-bootstrap";

const CodeEditor = ({disabledVal,codeText,}: {disabledVal: boolean; codeText: any;}) => {
    const [code, setCode] = React.useState(codeText || "Give any Input");
    const [disabled, setDisabled] = React.useState(true);

    const clearLocal = () => {
        if (localStorage.getItem("code")) {
            setCode("Give any Input");
            localStorage.clear();
        }
    };
    useEffect(() => {
        if (disabledVal) {
            setCode(localStorage.getItem("code") ||codeText || "Give any Input");
        }
        setDisabled(disabledVal);
    });

    return (
        <div style={{ margin: "20px" }}>
            <h2>Code Editor</h2>
            <Editor
                value={code}
                onValueChange={(code) => {setCode(code); localStorage.setItem("code", code); }}
                highlight={(code) => highlight(code, languages.py)}
                padding={40}
                style={{
                    fontFamily: '"Fira code", "Fira Mono", monospace',
                    fontSize: 20,
                    background: "#20202f",
                }}
                preClassName="edit"
                disabled={disabled}
            />
            <Button
                variant="danger"
                style={{ marginTop: "20px" }}
                onClick={clearLocal}
            >
                Clear code
      </Button>
        </div>
    );
};
export default CodeEditor;
