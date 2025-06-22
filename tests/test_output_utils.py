import pytest
import os
import sys
import pandas as pd
import json
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.d2d.utils.output_utils import (
    setup_logging,
    find_reference_for_answers,
    generate_output_from_summarized_matches_async
)

@pytest.fixture
def intermediate_objs():
    """
    Fixture to create intermediate processing objects to test output_utils.py functions

    The functions we are testing are at the end of the pipeline. Rather than call each 
        earlier method, we create test versions of the intermediate objects they will need
        in test fixtures (under `interview_004`). This function loads those objects from JSON
        and creates list/dict to be used as params
    Use similar structure to set up additional test cases for output_utils.py
    
    Returns:
        dict: containing keys:
            - transcript_files (list[str]): List of transcript file names from `transcripts_filenames.json`. Files not real.
            - guide_questions (list[str]): List of guidelines questions from file `guideline_questions.json`.
            - matches_list (list[list[dict]]): List of list of dictionaries with `guide_question` and `matches` from
                `matches_list.json`
    """

    test_case = "004"
    json_path = Path(__file__).parent / "fixtures" / f"interview_{test_case}"
    
    transcripts_file_path = json_path / "transcripts_filenames.json"
    with open(str(transcripts_file_path), "r") as f:
        data1 = json.load(f)

    guidelines_path = json_path / "guideline_questions.json"
    with open(str(guidelines_path), "r") as f:
        data2 = json.load(f)

    matches_list_path = json_path / "matches_list.json"
    with open(str(matches_list_path), "r") as f:
        data3 = json.load(f)
        
    return {
        "transcript_files": data1, 
        "guide_questions": data2, 
        "matches_list": data3
        }

## get_divider: Utility function, no tests

## output_divider: Utility function, no tests

## setup_logging: Utility function, no tests


## find_reference_for_answers

def test_find_references_simple(subtests, intermediate_objs, 
                                logger, embedding_model, torch_device):
    """
    Confirm that the correct reference details are derived from a simple test case
        'going great'
    """
    extracted_phrase = "going great"
    # retrieve all interview matches from fixture
    matches_list = intermediate_objs["matches_list"]
    # for this test case, we use only the first question (how are you feeling at work) 
    # of the 2nd interview (index 1)
    match = (
        next((obj for obj in matches_list[1] 
              if obj["guide_question"] == "How are you feeling at work?"), None)
              )
    (
        line_reference, response_position,
        interviewee_match, match_type
    ) = find_reference_for_answers(match, extracted_phrase, 
                                       embedding_model, torch_device, logger)
    
    # print(line_reference, response_position, interviewee_match, match_type)
    with subtests.test("Confirm line reference is as expected"):
        assert line_reference==[123]

    with subtests.test("Confirm line reference position is as expected"):
        assert len(response_position)==1
        assert response_position[0]['line']==123
        assert response_position[0]['start']==5
        assert response_position[0]['end']==16

    with subtests.test("Confirm reference text is as expected"):
        assert interviewee_match==["It's going great, actually"]

    with subtests.test("Confirm match type is as expected"):
        assert match_type=="EXACT"
    
def test_find_references_case_insensitive(subtests, intermediate_objs, 
                                logger, embedding_model, torch_device):
    """
    Confirm that the correct reference details are derived even if the derived phrase 
    is in a different/mixed case 
        'gO CaMPing'
    """
    extracted_phrase = "gO CaMPing"
    # retrieve all interview matches from fixture
    matches_list = intermediate_objs["matches_list"]
    # for this test case, we use only the 2nd question (summer plans) 
    # of the 2nd interview (index 1)
    match = (
        next((obj for obj in matches_list[1] 
              if obj["guide_question"] == "What plans do you have for summer?"), None)
              )
    (
        line_reference, response_position,
        interviewee_match, match_type
    ) = find_reference_for_answers(match, extracted_phrase, 
                                       embedding_model, torch_device, logger)
    # print(line_reference, response_position, interviewee_match, match_type)
    with subtests.test("Confirm line reference is as expected"):
        assert line_reference==[313]
    with subtests.test("Confirm line reference position is as expected"):
        assert len(response_position)==1
        assert response_position[0]['line']==313
        assert response_position[0]['start']==38
        assert response_position[0]['end']==48

    with subtests.test("Confirm reference text is as expected"):
        assert interviewee_match==["I am not sure yet, but we'll probably go camping"]

    with subtests.test("Confirm match type is as expected"):
        assert match_type=="EXACT"

def test_find_references_partial(subtests, intermediate_objs, 
                                logger, embedding_model, torch_device):
    """
    Confirm that the correct reference details are derived from a simple test case
        'going great'
    """
    extracted_phrase = "I am feeling overwhelmed"
    # retrieve all interview matches from fixture
    matches_list = intermediate_objs["matches_list"]
    # for this test case, we use only the first question (how are you feeling at work) 
    # of the 1st interview (index 0)
    match = (
        next((obj for obj in matches_list[0] 
              if obj["guide_question"] == "How are you feeling at work?"), None)
              )
    (
        line_reference, response_position,
        interviewee_match, match_type
    ) = find_reference_for_answers(match, extracted_phrase, 
                                       embedding_model, torch_device, logger)
    
    # print(line_reference, response_position, interviewee_match, match_type)
    
    with subtests.test("Confirm line reference position is as expected"):
        # the reference should be two parts of the same line 7
        assert len(response_position)==2

        # first portion
        assert response_position[0]['line']==7
        assert response_position[0]['start']==2
        assert response_position[0]['end']==6

        # second portion
        assert response_position[1]['line']==7
        assert response_position[1]['start']==12
        assert response_position[1]['end']==24

    with subtests.test("Confirm match type is PARTIAL"):
        assert match_type=="PARTIAL"

def test_find_references_semantic(subtests, intermediate_objs, 
                                logger, embedding_model, torch_device):
    """
    Confirm that the correct reference details are derived from a case matched using
        semantic similarity
    """
    extracted_phrase = "It is fantastic"
    # retrieve all interview matches from fixture
    matches_list = intermediate_objs["matches_list"]
    # for this test case, we use only the first question (how are you feeling at work) 
    # of the 2nd interview (index 1)
    match = (
        next((obj for obj in matches_list[1] 
              if obj["guide_question"] == "How are you feeling at work?"), None)
              )
    # set the embedding for the match
    for m in match["matches"]:
        m["response_embedding"] = embedding_model.encode(m["response"], convert_to_tensor=True,
                                                         device=torch_device)

    # override logging to console to allow output to be viewed
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Show INFO and above on the console
    console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    (
        line_reference, response_position,
        interviewee_match, match_type
    ) = find_reference_for_answers(match, extracted_phrase, 
                                       embedding_model, torch_device, logger)
    
    # print(line_reference, response_position, interviewee_match, match_type)
    
    with subtests.test("Confirm line reference position is as expected"):
        assert len(response_position)==1
        assert response_position[0]['line']==123
        assert response_position[0]['start']==-1
        assert response_position[0]['end']==-1

    with subtests.test("Confirm reference text is as expected"):
        assert interviewee_match==["It's going great, actually"]

    with subtests.test("Confirm match type is as expected"):
        assert match_type=="SEMANTIC"


def test_find_references_no_match(subtests, intermediate_objs, 
                                logger, embedding_model, torch_device):
    """
    Confirm that no matches are returned for a case of mismatch
    """
    extracted_phrase = "go camping" # this should not match "How are you feeling at work?"
    # retrieve all interview matches from fixture
    matches_list = intermediate_objs["matches_list"]
    # for this test case, we use only the first question (how are you feeling at work) 
    # of the 2nd interview (index 1)
    match = (
        next((obj for obj in matches_list[1] 
              if obj["guide_question"] == "How are you feeling at work?"), None)
              )
    # set the embedding for the match
    for m in match["matches"]:
        m["response_embedding"] = embedding_model.encode(m["response"], 
                                                         convert_to_tensor=True,
                                                         device=torch_device)

    # override logging to console to allow output to be viewed
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Show INFO and above on the console
    console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    (
        line_reference, response_position,
        interviewee_match, match_type
    ) = find_reference_for_answers(match, extracted_phrase, 
                                       embedding_model, torch_device, logger)
    
    # print(line_reference, response_position, interviewee_match, match_type)
    
    with subtests.test("Confirm line number is empty"):
        assert len(line_reference)==0
    
    with subtests.test("Confirm line reference is empty"):
        assert len(response_position)==0

    with subtests.test("Confirm reference text is empty"):
        assert len(interviewee_match)==0

    with subtests.test("Confirm match type is None"):
        assert not match_type

    

## generate_output_from_summarized_matches_async
@pytest.mark.asyncio
async def test_generate_output_from_summarized_matches_async_valid(subtests, tmp_path, intermediate_objs, logger, 
                                                                   embedding_model, torch_device):
    base_output_name = "result"
    output_folder = tmp_path / "output" 
    output_folder.mkdir()
    csv_filebase = output_folder / f"{base_output_name}.csv"
    # override logging to console to allow output to be viewed
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Show INFO and above on the console
    console_handler.setFormatter(formatter)
    #logger.addHandler(console_handler)
    
    # more set up
    llm_model = "gpt-4o-mini"

    await generate_output_from_summarized_matches_async(intermediate_objs["transcript_files"],
                                                        intermediate_objs["matches_list"],
                                                        intermediate_objs["guide_questions"],
                                                        llm_model=llm_model,
                                                        output_path=csv_filebase,
                                                        embedding_model=embedding_model,
                                                        device=torch_device,
                                                        logger=logger)

     # output folder check
    with subtests.test("Output folder exists"):
        assert output_folder.exists()
        
    # folder should not be empty
    with subtests.test("Output folder not empty"):
        if not any(output_folder.iterdir()):        
            raise AssertionError(f"{output_folder} is empty.")

    # check the folder contents
    for filename in os.listdir(str(output_folder)):
        file_path = os.path.join(str(output_folder), filename)
        # print(f"--- Contents of {filename} ---")
        # with open(file_path, 'r') as file:
        #     print(file.read())
        # print("\n\n") # separator

        # test the content in the CSV file produced in output
        if(filename.endswith(".csv")):
            # filename should start with the specified basename
            # print("Base name: ", os.path.basename(filename), "~", base_output_name)
            with subtests.test("Check that the output CSV file is named correctly"):
                assert os.path.basename(filename).startswith(base_output_name)

            df = pd.read_csv(file_path)
            # check that the columns come from the guidelines
            with subtests.test("Output CSV has required columns"):
                assert "Interview File" in df.columns
                assert "How are you feeling at work?" in df.columns
                assert "What plans do you have for summer?" in df.columns
            # check that the row for the sample interview is as expected
            test_row = df[df["Interview File"]=="e9353054-0797"]
            with subtests.test("Output CSV row has expected values"):
                assert test_row["How are you feeling at work?"].str.contains("great", case=False).all()
                assert test_row["What plans do you have for summer?"].str.contains("camping", case=False).all()

        if(filename.endswith(".json")):
            # print("Base name: ", os.path.basename(filename), "~", base_output_name)
            with subtests.test("Check that the output JSON file is named correctly"):
                assert os.path.basename(filename).startswith(f"{base_output_name}_references")

            with open(file_path) as out_file:
                data = json.load(out_file)
                with subtests.test("Output should have 2 interviews"):
                    assert len(data)==2
                interview2 = (
                    next((obj for obj in data if obj["interview"] == "e9353054-0797"), None)
                )
                # the result needs to be there
                assert interview2
                
                q1 = (
                    next((obj for obj in interview2["responses"] 
                          if obj["guide_question"] == "How are you feeling at work?"), None)
                )
                with subtests.test("Check that Q1 for Interview 2 response references correct line"):
                    assert q1["extracted_line_references"]==[123]
                q2 = (
                    next((obj for obj in interview2["responses"] 
                          if obj["guide_question"] == "What plans do you have for summer?"), None)
                )
                
                with subtests.test("Check that Q2 for Interview 2 response references correct line"):
                    assert q2["extracted_line_references"]==[313]

                
@pytest.mark.asyncio
async def test_generate_output_from_summarized_matches_async_empty(subtests, tmp_path, intermediate_objs, logger, 
                                                                   embedding_model, torch_device):
    base_output_name = "result"
    output_folder = tmp_path / "output" 
    output_folder.mkdir()
    csv_filebase = output_folder / f"{base_output_name}.csv"
    # override logging to console to allow output to be viewed
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Show INFO and above on the console
    console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)
    
    # more set up
    llm_model = "gpt-4o-mini"
    
    # We must have at least one value for rows (files), questions (columns)
    # if either of these lists is empty, there should be a KeyError
    with subtests.test("Param transcript_files cannot be empty"):
        with pytest.raises(KeyError):
            await generate_output_from_summarized_matches_async(transcript_files=[],
                                                        matches_list=intermediate_objs["matches_list"],
                                                        guide_questions=intermediate_objs["guide_questions"],                                                        
                                                        llm_model=llm_model,
                                                        output_path=csv_filebase,
                                                        embedding_model=embedding_model,
                                                        device=torch_device,
                                                        logger=logger)
            
    with subtests.test("Param matches_list cannot be empty"):    
        with pytest.raises(KeyError):
            await generate_output_from_summarized_matches_async(transcript_files=intermediate_objs["transcript_files"],
                                                        matches_list=[],
                                                        guide_questions=intermediate_objs["guide_questions"],                                                        
                                                        llm_model=llm_model,
                                                        output_path=csv_filebase,
                                                        embedding_model=embedding_model,
                                                        device=torch_device,
                                                        logger=logger)
