from dash import Dash, dcc, html, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os
import glob
import json

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
data_dir = os.path.join(root_dir, "data", "private_data")
output_dir = os.path.join(root_dir, "results")

app.layout = dbc.Container(
    [
        dcc.Store(id='file-trigger', data={'data': data_dir, 'output': output_dir}),
        html.H1('Dialogue2Data'),
        dbc.Row(
            [
                dbc.Col(dcc.Dropdown(id='sel-results'),md=6),
                 dbc.Col(dcc.Dropdown(id='sel-interview'),md=6)
            ]
        ),
        dbc.Row(
            dbc.Col(dcc.Dropdown(id='sel-question'),md=12),
            style={"margin-top": "15px"}
        ),
        dbc.Row(
            [
                html.Div(id='txt-extracted-phrase'),
                html.Br(),
                dbc.Card(id='transcript',
                         children=[dbc.CardBody(html.Pre(id='txt-interview', 
                                                         children="transcript...", 
                                                         className="mb-1",
                                                         style={"whiteSpace": "pre-wrap", "height": "450px", "overflowY": "auto"}
                                                         ))]
                         )
                
            ],
            style={"margin-top": "20px"}
            ),
    ]
)

@callback(
    Output('sel-results', 'options'),
    Input('file-trigger', 'data')
)
def list_results(data):
    option_list = []
    for result_file in glob.glob(os.path.join(data['output'], '*.json')):
        option_list.append({
            'label': os.path.basename(result_file),
            'value': result_file
            })
    return option_list

@callback(
    Output('sel-interview', 'options'),
    Input('file-trigger', 'data'),
    Input('sel-results', 'value')
)
def list_interviews(data, selected_file):
    option_list = []
    #print(f"File is {selected_file}")
    #print(f"Look into {data['data']}")
    if selected_file:
        # get the files listed
        with open(selected_file, "r") as f:
            results_data = json.load(f)
            interview_files = [m["interview"] for m in results_data]
            #print(interview_files)
            for interview in glob.glob(os.path.join(data['data'], '**', '*.txt'), recursive=True):
                curr_filename = os.path.basename(interview).split(".")[0]
                if curr_filename in interview_files:
                    option_list.append({
                        'label': os.path.basename(curr_filename),
                        'value': interview
                        })
            # debug: print any results without references
            all_null = []
            for m in results_data:
                null_refs = [{"q": r["guide_question"], "ext": r["extracted_phrase"]} for r in m["responses"] if not r["extracted_line_references"]]
                if len(null_refs)>0:
                    all_null.append((m["interview"], null_refs))
            print(f"NULLS: {json.dumps(all_null, indent=4)}")
    return option_list


@callback(
    Output('sel-question', 'options'),
    Input('sel-results', 'value'),
    Input('sel-interview', 'value')
)
def list_questions(results_file, selected_interview):
    option_list = []
    if selected_interview:
        with open(results_file, "r") as f:
            results_data = json.load(f)
            curr_filename = os.path.basename(selected_interview).split(".")[0]
            curr_interview = next((item["responses"] for item in results_data 
                          if item["interview"] == curr_filename), None)
            if curr_interview:
                option_list = [item["guide_question"] for item in curr_interview]
                #print(option_list)
            #print(selected_interview)
            #print(curr_interview)
    return option_list

@callback(
    Output('txt-interview', 'children'),
    Output('txt-extracted-phrase', 'children'),
    Input('sel-interview', 'value'),
    Input('sel-results', 'value'),
    Input('sel-question', 'value')
)
def render_interview(selected_interview, results_file, question):    
    if selected_interview:
        transcript = []
        extracted_phrase =  html.Span()
        with open(selected_interview, "r") as f:
            transcript_data = f.read()
        # if a question is selected, show its matches
        if question:
            with open(results_file, "r") as f:
                results_data = json.load(f)
                curr_filename = os.path.basename(selected_interview).split(".")[0]
                curr_interview = next((item["responses"] for item in results_data 
                            if item["interview"] == curr_filename), None)
                question = next((item for item in curr_interview 
                            if item["guide_question"] == question), None)
                
                extracted_phrase =  [dbc.Alert(f"Extracted Phrase: {question["extracted_phrase"]}", color="primary"),
                                     dbc.Alert(f"Response: {question["response"]}", color="success")]
                # interviewer questions
                relevant_qn_lines = [q[0] for q in question["relevant_lines"]]
                # interviewee responses
                relevant_resp_lines = [q[1] for q in question["relevant_lines"]]
                # specific segments in response
                segments = question["extracted_character_index"]
                #print(f"Interviewer: {relevant_qn_lines}")
                #print(f"Interviewee: {relevant_resp_lines}")
                print(f"Segments: {segments}")
                lines = transcript_data.split(sep='\n', )
                for line_number, line in enumerate(lines, start=1):
                    if line_number in relevant_resp_lines:
                        # check if specific sections are to be highlighted
                        if segments:
                            sections = [s for s in segments if s["line"]==line_number]
                            sections = sorted(sections, key=lambda x: x['start'])
                        else:
                            sections = []
                        print(f"Line {line_number}; mark sections: {sections}")
                        # check if specific index is mentioned
                        if (any([s for s in sections if s["start"]>=0])):
                            # mark specific sections
                            cursor = 0
                            for s in sections:
                                # we have to adjust for the length of "Interviewer: "/"Interviewee: "
                                adjust_len = len("Interviewer: ")
                                start, end = s["start"] + adjust_len, s["end"] + adjust_len
                                if start > cursor:
                                    # the part before the match begin
                                    transcript.append(html.Span(line[cursor:start], style={"backgroundColor": "yellow"}))  
                                transcript.append(html.Span(line[start:end],
                                                            style={"backgroundColor": "yellow", "fontWeight": "bold"}
                                                            ))
                                cursor = end

                                # text after last match
                                if cursor < len(line):
                                    transcript.append(html.Span(line[cursor:], style={"backgroundColor": "yellow"}))
                                #add end of line
                                transcript.append(html.Span("\n", style={"backgroundColor": "yellow"}))
                        elif len(sections)>0:
                            #mark whole line
                            transcript.append(html.Span(f"{line}\n", style={"backgroundColor": "yellow", "fontWeight": "bold"}))
                        else:
                            # line was just relevant, but not used for retrieval                            
                            transcript.append(html.Span(f"{line}\n", style={"backgroundColor": "yellow"}))
                            
                    elif line_number in relevant_qn_lines:
                        transcript.append(html.Span(f"{line}\n", style={"backgroundColor": "yellow", "fontStyle": "italic"}))
                    else:
                        transcript.append(html.Span(f"{line}\n"))
        else:
            transcript = transcript_data
        return transcript, extracted_phrase
    else:
        return "Select an interview...", ""

if __name__ == '__main__':
    app.run(debug=True)