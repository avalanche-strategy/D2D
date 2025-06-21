import pytest
import os
import sys
import torch
from sentence_transformers import SentenceTransformer, util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.d2d.utils.data_utils import load_guidelines, load_transcript, segment_transcript
from src.d2d.utils.embedding_utils import (
    summarize_embed_groups_async,
    summarize_match_top_k_questions_async,
    summarize_match_top_p_questions_async,
    match_top_responses
)

## summarize_embed_groups_async

@pytest.mark.asyncio
async def test_summarize_embed_groups_async_valid(subtests, test_case_files, logger):
    """
    Tests that loading a valid interview file works as expected.

    Args:
        test_case_files (fixture function): A fixture function that create a
            temporary pathlib.Path from fixtures directory
        logger : logging fixture
    """

    test_case = "000"
    test_transcript_file = "fe4b129c-a507.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    interview_data = segment_transcript(interview_file_content)
    
    # In this test, we use a custom SentenceTransformer with lower dim embeddings
    sentence_model_name = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(sentence_model_name)
    torch_device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
        )
    llm_model = "gpt-4o-mini"

    result = await summarize_embed_groups_async(interview_data, 
                                                embedding_model, torch_device, 
                                                llm_model, logger)
    # check the output maintains the 2 QA pairs
    assert len(result)==2

    # # properties maintained
    # {'interviewee_line_ref', 'interviewer_line_ref', 'speaking_round'}

    # # new properties added
    # {'embedding', 'response_embedding', 'original_question', 'summarized_question', 'original_response'}
    
    # # properties removed
    # {"interviewer", "interviewee"}

    # check that the output object has all required keys    
    expected_keys = {'original_question', 'embedding', 'interviewee_line_ref', 
                     'summarized_question', 'response_embedding', 'original_response', 
                     'speaking_round', 'interviewer_line_ref'}
    result_keys = set([k for r in result for k in r.keys()])
    
    with subtests.test(f"The results have all expected keys"):
        assert expected_keys.difference(result_keys) == set()

    # we cannot test embedding values, but we check dimension is 384 (all-MiniLM-L6-v2)
    with subtests.test(f"The embedding for all items is a 384 dimension tensor"):
        for elem in result:
            assert isinstance(elem['embedding'], torch.Tensor)
            assert elem['embedding'].shape[0]==384
#             assert elem['response_embedding'].shape[0]==384    

@pytest.mark.asyncio
async def test_summarize_embed_groups_async_empty(logger, embedding_model, torch_device):
    """
    Tests that loading an empty list should work as expected

    Args:
        logger : logging fixture
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """
    
    interview_data = []
    
    # more set up
    llm_model = "gpt-4o-mini"

    result = await summarize_embed_groups_async(interview_data, 
                                                embedding_model, torch_device, 
                                                llm_model, logger)
    # check the output returns an empty list, no exception
    assert len(result)==0

@pytest.mark.asyncio
async def test_summarize_embed_groups_async_interviewer_missing(logger, embedding_model, torch_device):
    """
    Tests that loading a list without "interviewer"/"interviewee" generates missing key erro

    Args:
        logger : logging fixture
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """
    
    interview_data = [{
        "intervieweee": ["Whatever..."]
    }]
    
    # more set up
    llm_model = "gpt-4o-mini"

    with pytest.raises(KeyError):
        await summarize_embed_groups_async(interview_data,
                                           embedding_model, torch_device,
                                           llm_model, logger)
    

# ## summarize_match_top_k_questions_async
@pytest.mark.asyncio
async def test_summarize_match_top_k_questions_async_valid_k1(subtests, test_case_files, logger,
                                                              embedding_model, torch_device):
    """
    Tests that choosing top_k=1 picks the expected line with highest similarity

    Args:
        test_case_files (fixture function): A fixture function that create a
            temporary pathlib.Path from fixtures directory
        logger : logging fixture
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """

    test_case = "000"
    test_transcript_file = "fe4b129c-a507.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    interview_data = segment_transcript(interview_file_content)
    
    # more set up
    llm_model = "gpt-4o-mini"
    guide_embedding = embedding_model.encode("How is it at work?", convert_to_tensor=True, device=torch_device)

    embedded_groups = await summarize_embed_groups_async(interview_data,
                                                         embedding_model, torch_device,
                                                         llm_model, logger)
    result = await summarize_match_top_k_questions_async(guide_embedding, embedded_groups, k=1)

    # we should only have 1 result
    assert len(result)==1
    # confirm that the correct interviewer line was retrieved
    with subtests.test("Confirm 'interviewer' line reference for k=1"):
        assert result[0]['interviewer_line_ref']==5

    # confirm that the correct interviewee line was retrieved
    with subtests.test("Confirm 'interviewee' line reference for k=1"):
        assert result[0]['interviewee_line_ref']==7

@pytest.mark.asyncio
async def test_summarize_match_top_k_questions_async_valid_k2(subtests, test_case_files, logger,
                                                              embedding_model, torch_device):
    """
    Tests that choosing top_k=2 picks both lines, including 2nd line with low similarity

    Args:
        test_case_files (fixture function): A fixture function that create a
            temporary pathlib.Path from fixtures directory
        logger : logging fixture
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """

    test_case = "000"
    test_transcript_file = "fe4b129c-a507.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    interview_data = segment_transcript(interview_file_content)
    
    # more set up
    llm_model = "gpt-4o-mini"
    guide_embedding = embedding_model.encode("How are things at work?", 
                                             convert_to_tensor=True, device=torch_device)

    embedded_groups = await summarize_embed_groups_async(interview_data,
                                                         embedding_model, torch_device,
                                                         llm_model, logger)

    with subtests.test("Confirm K=2 returns all 2 items"):
        result = await summarize_match_top_k_questions_async(guide_embedding, embedded_groups, k=2)
        # print([round(i['similarity'],4) for i in result])
        # we should only have 2 results
        assert len(result)==2
    
    with subtests.test("Confirm K=5 returns just the 2 available items"):
        result = await summarize_match_top_k_questions_async(guide_embedding, embedded_groups, k=5)
        # print([round(i['similarity'], 4) for i in result])
        # we should only have 2 results
        assert len(result)==2

@pytest.mark.asyncio
async def test_summarize_match_top_k_questions_async_empty(embedding_model, torch_device):
    """
    Tests that passing an empty embeddings groups returns empty list for top k matching

    Args:
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """
    
    guide_embedding = embedding_model.encode("How are things at work?", 
                                             convert_to_tensor=True, device=torch_device)

    result = await summarize_match_top_k_questions_async(guide_embedding, group_embeddings=[], k=2)

    # check the output returns an empty list, no exception
    assert len(result)==0


@pytest.mark.asyncio
async def test_summarize_match_top_k_questions_async_manual(subtests, test_case_files, logger,
                                                            embedding_model, torch_device):
    """
    Tests that choosing top_k=1 picks the expected result with expected similarity

    Args:
        test_case_files (fixture function): A fixture function that create a
            temporary pathlib.Path from fixtures directory
        logger : logging fixture
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """

    test_case = "000"
    test_transcript_file = "fe4b129c-a507.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    interview_data = segment_transcript(interview_file_content)
    
    # more set up
    llm_model = "gpt-4o-mini"
    guide_embedding = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32).to(device=torch_device)

    embedded_groups = await summarize_embed_groups_async(interview_data,
                                                         embedding_model, torch_device,
                                                         llm_model, logger)
    # modify the embedding
    # ~ 0.999 cosine
    embedded_groups[0]['embedding'] = torch.tensor([0.371,0.557,0.743], dtype=torch.float32).to(device=torch_device)
    # ~ 0.1 cosine
    embedded_groups[1]['embedding'] = torch.tensor([0.867, 0.207, -0.454], dtype=torch.float32).to(device=torch_device)    
    result = await summarize_match_top_k_questions_async(guide_embedding, embedded_groups, k=5)
    # confirm result 1
    with subtests.test("Confirm Embedding with similarity ~0.999 is first"):
        assert torch.allclose(torch.tensor([result[0]['similarity']]), 
                            torch.tensor([0.999]),
                            atol=0.001, equal_nan=False)
        # manually setting the embedding forced this line to have a high similarity, 
        # even though it is different semantically
        assert result[0]['interviewer_line_ref']==1
        assert result[0]['interviewee_line_ref']==3
    # confirm result 2
    with subtests.test("Confirm Embedding with similarity ~0.1 is second"):
        assert torch.allclose(torch.tensor([result[1]['similarity']]),
                            torch.tensor([0.1]), 
                            atol=0.001, equal_nan=False)
        # manually setting the embedding forced this line to have a low similiarity, 
        # even though semantically closer to the question
        assert result[1]['interviewer_line_ref']==5
        assert result[1]['interviewee_line_ref']==7



## summarize_match_top_p_questions_async

@pytest.mark.asyncio
async def test_summarize_match_top_p_questions_async_valid_p_0_5(subtests, test_case_files, logger, 
                                                                 embedding_model, torch_device):
    """
    Tests that choosing top_p=0.5 picks the expected line with highest similarity

    Args:
        test_case_files (fixture function): A fixture function that create a
            temporary pathlib.Path from fixtures directory
        logger : logging fixture
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """

    test_case = "000"
    test_transcript_file = "fe4b129c-a507.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    interview_data = segment_transcript(interview_file_content)
    
    # variables
    llm_model = "gpt-4o-mini"
    guide_embedding = embedding_model.encode("How is it at work?", convert_to_tensor=True, device=torch_device)

    embedded_groups = await summarize_embed_groups_async(interview_data,
                                                         embedding_model, torch_device,
                                                         llm_model, logger)
    result = await summarize_match_top_p_questions_async(guide_embedding, embedded_groups, p=0.5)

    # we should only have 1 result
    assert len(result)==1
    # confirm that the correct interviewer line was retrieved
    with subtests.test("Confirm 'interviewer' line reference for top p=0.5"):
        assert result[0]['interviewer_line_ref']==5

    # confirm that the correct interviewee line was retrieved
    with subtests.test("Confirm 'interviewee' line reference for top p=0.5"):
        assert result[0]['interviewee_line_ref']==7

@pytest.mark.asyncio
async def test_summarize_match_top_p_questions_async_valid_p_high_low(subtests, test_case_files, logger,
                                                                      embedding_model, torch_device):
    """
    Tests that choosing top_p=0.999 picks nothing and top_p=0.001 picks both lines

    Args:
        test_case_files (fixture function): A fixture function that create a
            temporary pathlib.Path from fixtures directory
        logger : logging fixture
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """

    test_case = "000"
    test_transcript_file = "fe4b129c-a507.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    interview_data = segment_transcript(interview_file_content)
    
    # more set up
    llm_model = "gpt-4o-mini"
    guide_embedding = embedding_model.encode("How is it at work?", convert_to_tensor=True, device=torch_device)

    embedded_groups = await summarize_embed_groups_async(interview_data,
                                                         embedding_model, torch_device,
                                                         llm_model, logger)
    
    # confirm that a very high similarity threshold returns empty list
    with subtests.test("Match TOP_P=0.999 has 0 matches"):
        result = await summarize_match_top_p_questions_async(guide_embedding, embedded_groups, p=0.999)
        # we should only have 0 results
        assert len(result)==0

    # confirm that a very low similarity threshold returns everything
    with subtests.test("Match TOP_P=0.001 has 2 matches"):
        result = await summarize_match_top_p_questions_async(guide_embedding, embedded_groups, p=0.001)
        # we should only have 2 results
        assert len(result)==2

@pytest.mark.asyncio
async def test_summarize_match_top_p_questions_async_empty(embedding_model, torch_device):
    """
    Tests that passing an empty embeddings groups returns empty list for top p matching

    Args:
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """
    
    guide_embedding = embedding_model.encode("How are things at work?", 
                                             convert_to_tensor=True, device=torch_device)

    result = await summarize_match_top_p_questions_async(guide_embedding, group_embeddings=[], p=0.5)

    # check the output returns an empty list, no exception
    assert len(result)==0


@pytest.mark.asyncio
async def test_summarize_match_top_p_questions_async_manual(subtests, test_case_files, logger,
                                                            embedding_model, torch_device):
    """
    Tests that choosing top_p=0.5 picks the expected result with expected similarity value
    for manually set embedding vectors

    Args:
        test_case_files (fixture function): A fixture function that create a
            temporary pathlib.Path from fixtures directory
        logger : logging fixture
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """

    test_case = "000"
    test_transcript_file = "fe4b129c-a507.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    interview_data = segment_transcript(interview_file_content)
    
    # more set up
    llm_model = "gpt-4o-mini"
    guide_embedding = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32).to(device=torch_device)

    embedded_groups = await summarize_embed_groups_async(interview_data,
                                                         embedding_model, torch_device,
                                                         llm_model, logger)
    # modify the embedding
    # ~ 0.999 cosine
    embedded_groups[0]['embedding'] = torch.tensor([0.371,0.557,0.743], dtype=torch.float32).to(device=torch_device)
    # ~ 0.1 cosine
    embedded_groups[1]['embedding'] = torch.tensor([0.867, 0.207, -0.454], dtype=torch.float32).to(device=torch_device)    

    
    # confirm result 1
    with subtests.test("Confirm only 1 result for p=0.5 for manually set embeddings"):
        result = await summarize_match_top_p_questions_async(guide_embedding, embedded_groups, p=0.5)
        assert len(result)==1
        assert torch.allclose(torch.tensor([result[0]['similarity']]), 
                            torch.tensor([0.999]),
                            atol=0.001, equal_nan=False)
        # confirm that manually setting the embedding value returned the first line only
        assert result[0]['interviewer_line_ref']==1
        assert result[0]['interviewee_line_ref']==3

    # confirm result 2
    with subtests.test("Confirm there are 2 results for p=0.6 with manually set embeddings"):
        # and now, change the second group to target a similarity of ~0.75
        embedded_groups[1]['embedding'] = torch.tensor([0.83, 0.52, 0.206], dtype=torch.float32).to(device=torch_device)
        result = await summarize_match_top_p_questions_async(guide_embedding, embedded_groups, p=0.6)
        
        assert len(result)==2
        assert torch.allclose(torch.tensor([result[1]['similarity']]),
                            torch.tensor([0.75]), 
                            atol=0.001, equal_nan=False)
        # manually setting the embedding forced this line to have the lower similiarity value, 
        # even though more semantically related
        assert result[1]['interviewer_line_ref']==5
        assert result[1]['interviewee_line_ref']==7


## match_top_responses

@pytest.mark.asyncio
async def test_match_top_responses_valid(subtests, test_case_files, logger, embedding_model, torch_device):
    """
    Tests response lines from Interview 000 fixture matches the first line

    Args:
        test_case_files (fixture function): A fixture function that create a
            temporary pathlib.Path from fixtures directory
        logger : logging fixture
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """

    test_case = "000"
    test_transcript_file = "fe4b129c-a507.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    interview_data = segment_transcript(interview_file_content)
    
    # more set up
    llm_model = "gpt-4o-mini"
    extracted_phrase = "No problem!"

    embedded_groups = await summarize_embed_groups_async(interview_data,
                                                         embedding_model, torch_device,
                                                         llm_model, logger)
    # mock responses from LLM, echoed back
    embedded_groups[0]['response'] = "Response 0"
    embedded_groups[1]['response'] = "Response 1"
    # test top responses referencing

    result = match_top_responses(embedding_model, torch_device, logger,
                                 extracted_phrase, embedded_groups, p_threshold=0.8)

    
    # we should only have 1 result
    assert len(result)==1

    # confirm that the correct interviewer line was retrieved
    with subtests.test("Confirm 'interviewee' line reference matched was correct"):
        assert result[0]['interviewee_line_ref']==3
        assert result[0]['response']=="Response 0"


@pytest.mark.asyncio
async def test_match_top_responses_invalid(test_case_files, logger, embedding_model, torch_device):
    """
    Tests that an invalid response will not be matched to any lines

    Args:
        test_case_files (fixture function): A fixture function that create a
            temporary pathlib.Path from fixtures directory
        logger : logging fixture
        embedding_model: SentenceTransformer embedding model fixture
        torch_device: torch.device fixture
    """

    test_case = "000"
    test_transcript_file = "fe4b129c-a507.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    interview_data = segment_transcript(interview_file_content)
    
    # more set up
    llm_model = "gpt-4o-mini"
    extracted_phrase = "Just imagine that the LLM was hallucinating here."

    embedded_groups = await summarize_embed_groups_async(interview_data,
                                                         embedding_model, torch_device,
                                                         llm_model, logger)
    # mock responses from LLM, echoed back
    embedded_groups[0]['response'] = "Response 0"
    embedded_groups[1]['response'] = "Response 1"
    # test top responses referencing

    result = match_top_responses(embedding_model, torch_device, logger,
                                 extracted_phrase, embedded_groups, p_threshold=0.8)

    # this one should not match any line 
    assert len(result)==0
    
