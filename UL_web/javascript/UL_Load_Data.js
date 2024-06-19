import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js"

function dataUpload(node, inputName, inputData, app) {
    const dataWidget = node.widgets.find((w) => w.name === "data");
    let uploadWidget;
    /* 
    A method that returns the required style for the html 
    */
    var default_value = dataWidget.value;
    Object.defineProperty(dataWidget, "value", {
        set: function (value) {
            this._real_value = value;
        },

        get: function () {
            let value = "";
            if (this._real_value) {
                value = this._real_value;
            } else {
                return default_value;
            }

            if (value.filename) {
                let real_value = value;
                value = "";
                if (real_value.subfolder) {
                    value = real_value.subfolder + "/";
                }

                value += real_value.filename;

                if (real_value.type && real_value.type !== "input")
                    value += ` [${real_value.type}]`;
            }
            return value;
        }
    });
    async function uploadFile(file, updateNode, pasted = false) {
        try {
            // Wrap file in formdata so it includes filename
            const body = new FormData();
            body.append("image", file);
            if (pasted) body.append("subfolder", "pasted");
            const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body,
            });

            if (resp.status === 200) {
                const data = await resp.json();
                // Add the file to the dropdown list and update the widget value
                let path = data.name;
                if (data.subfolder) path = data.subfolder + "/" + path;

                if (!dataWidget.options.values.includes(path)) {
                    dataWidget.options.values.push(path);
                }

                if (updateNode) {
                    dataWidget.value = path;
                }
            } else {
                alert(resp.status + " - " + resp.statusText);
            }
        } catch (error) {
            alert(error);
        }
    }

    const fileInput = document.createElement("input");
    Object.assign(fileInput, {
        type: "file",
        accept: "file/pdf,file/txt",
        style: "display: none",
        onchange: async () => {
            if (fileInput.files.length) {
                await uploadFile(fileInput.files[0], true);
            }
        },
    });
    document.body.append(fileInput);

    // Create the button widget for selecting the files
    uploadWidget = node.addWidget("button", "choose data file to upload", "Audio", () => {
        fileInput.click();
    });

    uploadWidget.serialize = false;

    const cb = node.callback;
    dataWidget.callback = function () {
        if (cb) {
            return cb.apply(this, arguments);
        }
    };

    return { widget: uploadWidget };
}

ComfyWidgets.DATAUPLOAD = dataUpload;

app.registerExtension({
    name: "UL.UL_Load_Data",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name == "UL_Load_Data") {
            nodeData.input.required.upload = ["DATAUPLOAD"];
        }
    },
});

