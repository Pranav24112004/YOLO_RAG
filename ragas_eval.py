"""
ragas_eval.py
=============
Evaluates the YOLO RAG agent using RAGAS metrics.

Flow:
  1. Define golden evaluation dataset (user_input + reference).
  2. For each question, invoke the LangGraph agent to get:
       - retrieved_contexts (documents from Azure AI Search)
       - response           (LLM-generated answer)
  3. Build a HuggingFace Dataset using native RAGAS v0.2+ keys.
  4. Run evaluate() with RunConfig throttling.
  5. Print per-question scores + aggregated metrics.
"""

import os
import sys
import json
import nest_asyncio
import pandas as pd
from datasets import Dataset

# Patch async event loop for local/Windows environments
nest_asyncio.apply()

# ── Import your existing agent ────────────────────────────────────────────────
from agent.graph import app as agent_app
from services.azure_openai import get_chat_llm
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import config.settings as settings

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
# ══════════════════════════════════════════════════════════════════════════════
# 1. GOLDEN EVALUATION DATASET
# ══════════════════════════════════════════════════════════════════════════════

EVAL_QUESTIONS = [
    "Describe the class prediction mechanism used in YOLOv3 and explain why softmax is not used.",
    "What is Darknet-53 and how does it compare with ResNet-101 and ResNet-152 according to the paper YOLOv3?",
    "Explain the additional improvements introduced in YOLOv4, specifically Mosaic augmentation and Self-Adversarial Training (SAT).",
    "How does YOLOv6 compare with other YOLO-series detectors in terms of speed and accuracy under the same testing conditions?",
    "How does YOLO fundamentally change the architecture of object detection compared to prior systems like R-CNN or DPM, and what is the primary advantage of this new approach?",
    "Fast R-CNN is a highly accurate detection method. How does YOLO's global reasoning give it an advantage over Fast R-CNN when it comes to background errors?",
    "The YOLOv4 paper emphasizes accessibility and practical application for everyday researchers. How does YOLOv4 achieve this, and what is the main hardware advantage it offers compared to other highly accurate modern neural networks?",
    "YOLOv4 categorizes its architectural and training improvements into 'Bag of Freebies' and 'Bag of Specials'. Can you explain the fundamental difference between these two categories and describe the two novel data augmentation techniques introduced as 'Bag of Freebies' in this paper?",
    "The YOLOv4 paper discusses how the optimal reference model for classification isn't necessarily optimal for object detection. According to the YOLOv4 developers, what specific architectural characteristics does a detector require compared to a classifier, and how did this influence their choice between CSPResNeXt50 and CSPDarknet53?",
    "In YOLOv4 expanded list of 'Bag of Freebies' for the detector, the authors mention a technique to 'eliminate grid sensitivity.' What specific issue from YOLOv3 does this address, and how does YOLOv4 solve it?",
    "In YOLOv3, the architecture deliberately moves away from using a softmax function for class prediction. What method does it use instead, and why is this architectural change particularly beneficial for complex datasets?",
    "The YOLOv3 paper introduces a new feature extractor network called Darknet-53. How does this new architecture differ from its predecessor, Darknet-19, and how does its efficiency compare to competing ResNet models?",
    "Previous versions of YOLO famously struggled with detecting small objects. How does YOLOv3 alter its prediction architecture to solve this, and what unexpected trade-off occurred as a result?",
    "When training the bounding box regression, traditional object detectors use Mean Square Error (MSE) to directly estimate coordinates. Why does the YOLOv4 paper consider this problematic, and how do improvements like CIoU loss address this issue?",
    "Activation functions are a key focus in the deep learning research discussed in the paper. Why did the YOLOv4 developers specifically choose to exclude PReLU, SELU, and ReLU6 from their candidate list of activation functions for training?",
]

GROUND_TRUTHS = [
    (
        "Each box predicts the classes the bounding box may contain using multilabel classification. "
        "We do not use a softmax as we have found it is unnecessary for good performance, instead we "
        "simply use independent logistic classifiers. During training we use binary cross-entropy loss "
        "for the class predictions.\n"
        "This formulation helps when we move to more complex domains like the Open Images Dataset. "
        "In this dataset there are many overlapping labels (i.e. Woman and Person). Using a softmax "
        "imposes the assumption that each box has exactly one class which is often not the case. "
        "A multilabel approach better models the data."
    ),
    (
        "We use a new network for performing feature extraction. Our new network is a hybrid approach "
        "between the network used in YOLOv2, Darknet-19, and that newfangled residual network stuff. "
        "Our network uses successive 3x3 and 1x1 convolutional layers but now has some shortcut "
        "connections as well and is significantly larger. It has 53 convolutional layers so we call it "
        "Darknet-53.\n"
        "This new network is much more powerful than Darknet-19 but still more efficient than ResNet-101 "
        "or ResNet-152.\n"
        "Darknet-53 performs on par with state-of-the-art classifiers but with fewer floating point "
        "operations and more speed. Darknet-53 is better than ResNet-101 and 1.5x faster. Darknet-53 "
        "has similar performance to ResNet-152 and is 2x faster.\n"
        "Darknet-53 also achieves the highest measured floating point operations per second. This means "
        "the network structure better utilizes the GPU, making it more efficient to evaluate and thus faster. "
        "That's mostly because ResNets have just way too many layers and aren't very efficient."
    ),
    (
        "Mosaic represents a new data augmentation method that mixes 4 training images. Thus 4 different "
        "contexts are mixed, while CutMix mixes only 2 input images. This allows detection of objects "
        "outside their normal context. In addition, batch normalization calculates activation statistics "
        "from 4 different images on each layer. This significantly reduces the need for a large mini-batch "
        "size.\n"
        "Self-Adversarial Training (SAT) also represents a new data augmentation technique that operates "
        "in 2 forward backward stages. In the 1st stage the neural network alters the original image "
        "instead of the network weights. In this way the neural network executes an adversarial attack "
        "on itself, altering the original image to create the deception that there is no desired object "
        "on the image. In the 2nd stage, the neural network is trained to detect an object on this "
        "modified image in the normal way.\n"
        "CmBN represents a CBN modified version, defined as Cross mini-Batch Normalization (CmBN). "
        "This collects statistics only between mini-batches within a single batch."
    ),
    (
        "Considering that the goal of this work is to build networks for industrial applications, "
        "we primarily focus on the speed performance of all models after deployment, including "
        "throughput (FPS at a batch size of 1 or 32) and the GPU latency, rather than FLOPs or the "
        "number of parameters. We compare YOLOv6 with other state-of-the-art detectors of YOLO series, "
        "including YOLOv5, YOLOX, PPYOLOE and YOLOv7. Note that we test the speed performance of all "
        "official models with FP16-precision on the same Tesla T4 GPU with TensorRT.\n"
        "Compared with YOLOv5-N/YOLOv7-Tiny (input size=416), our YOLOv6-N has significantly advanced "
        "by 7.9%/2.6% respectively. It also comes with the best speed performance in terms of both "
        "throughput and latency.\n"
        "Compared with YOLOX-S/PPYOLOE-S, YOLOv6-S can improve AP by 3.0%/0.4% with higher speed.\n"
        "YOLOv6-M outperforms YOLOv5-M by 4.2% higher AP with a similar speed, and it achieves "
        "2.7%/0.6% higher AP than YOLOX-M/PPYOLOE-M at a higher speed. Besides, it is more accurate "
        "and faster than YOLOv5-L.\n"
        "YOLOv6-L is 2.8%/1.1% more accurate than YOLOX-L/PPYOLOE-L under the same latency constraint."
    ),
    (
        "Prior systems repurposed classifiers to perform detection, using techniques like the sliding window approach "
        "in deformable parts models (DPM) or region proposal methods in R-CNN, which generate potential bounding boxes "
        "and then run a classifier on those specific boxes. These complex pipelines are slow and difficult to optimize "
        "because each individual component must be trained separately. YOLO, on the other hand, reframes object detection "
        "as a single regression problem, predicting bounding box coordinates and class probabilities straight from "
        "image pixels. A single convolutional neural network evaluates the full image at once to predict multiple "
        "bounding boxes and their class probabilities simultaneously. The primary advantage of this unified architecture "
        "is that it is extremely fast, with the base YOLO model processing images in real-time at 45 frames per second, "
        "and Fast YOLO processing them at 155 frames per second."
    ),
    (
        "Unlike region proposal-based techniques, YOLO sees the entire image during both training and test time, "
        "allowing it to implicitly encode contextual information about classes and their appearance. Fast R-CNN "
        "cannot see this larger context, causing it to frequently mistake background patches in an image for "
        "actual objects. Specifically, 13.6% of Fast R-CNN's top detections are false positives that do not "
        "contain any objects, making it almost three times more likely to predict background detections than YOLO. "
        "Because YOLO makes far fewer background mistakes, researchers found that combining YOLO with Fast R-CNN "
        "by using YOLO to eliminate Fast R-CNN's background detections provided a significant performance boost."
    ),
    (
        "The authors of YOLOv4 address the issue that most highly accurate modern neural networks cannot operate in real time "
        "and require a large number of GPUs for training with a large mini-batch size. They overcome this by developing an "
        "efficient and powerful CNN that not only operates in real-time on a conventional GPU but also only requires a single "
        "conventional GPU (such as a 1080 Ti or 2080 Ti) for training. To ensure the model remains accessible for single-GPU "
        "training, the developers deliberately modified existing state-of-the-art methods and avoided using expensive "
        "specialized devices or multi-GPU optimizations like Cross-GPU Batch Normalization (SyncBN). This design allows anyone "
        "to achieve state-of-the-art, real-time object detection results on standard, affordable hardware."
    ),
    (
        "The fundamental difference lies in their impact on inference cost. 'Bag of freebies' are methods that improve the "
        "object detector's accuracy without increasing the inference cost at all, typically by only altering the off-line "
        "training strategy or increasing training cost. In contrast, 'Bag of specials' are plugin modules or post-processing "
        "methods that do slightly increase the inference cost, but provide a significant improvement to accuracy by enhancing "
        "attributes like the receptive field or attention mechanisms.\n"
        "The researchers introduced two new 'Bag of Freebies' data augmentation methods:\n"
        "1. Mosaic: This method mixes four different training images together, allowing the model to learn to detect objects "
        "outside of their normal context. Because it processes four different images simultaneously on each layer, it also "
        "allows batch normalization to calculate activation statistics across a wider variety of inputs, significantly reducing "
        "the need for a large mini-batch size.\n"
        "2. Self-Adversarial Training (SAT): This operates in two forward-backward stages. In the first stage, the neural network "
        "executes an adversarial attack on itself by altering the original image to deceive the network into thinking there is "
        "no desired object. In the second stage, the network is trained normally to detect the object on this newly modified image."
    ),
    (
        "According to the authors, a detector requires a higher input network resolution to detect multiple small-sized objects, "
        "more layers to create a larger receptive field that covers the increased input size, and more parameters to give the model "
        "greater capacity to detect multiple objects of various sizes within a single image. While their studies demonstrated "
        "that CSPResNext50 performs considerably better for object classification on the ImageNet dataset, CSPDarknet53 is superior "
        "for object detection on the MS COCO dataset. This is because CSPDarknet53 possesses a larger 725 × 725 receptive field and "
        "27.6 million parameters, compared to CSPResNext50's smaller 425 × 425 receptive field and 20.6 million parameters. This "
        "theoretical justification, combined with their experimental results, led them to choose CSPDarknet53 as the optimal backbone model."
    ),
    (
        "In YOLOv3, the evaluation of object coordinates uses the equations bx = σ(tx) + cx and by = σ(ty) + cy, where cx and cy "
        "are strictly whole numbers. Because of how the sigmoid function (σ) works, the model requires extremely high absolute "
        "values for tx in order for the predicted coordinate bx to approach the whole numbers cx or cx + 1. This makes it very "
        "difficult for the model to detect objects that fall exactly on the grid boundaries. YOLOv4 solves this issue simply "
        "by multiplying the sigmoid by a factor exceeding 1.0. This adjustment eliminates the grid sensitivity effect, allowing "
        "the model to more easily predict coordinates that lie on the grid lines where an object would otherwise be undetectable."
    ),
    (
        "Instead of a softmax function, YOLOv3 uses independent logistic classifiers to perform multilabel classification. "
        "The developers made this change because a softmax function imposes the strict assumption that each bounding box "
        "contains exactly one class. This assumption fails in complex domains like the Open Images Dataset, which contains "
        "many overlapping labels, such as a single object being labeled as both 'Woman' and 'Person'. By using a multilabel "
        "approach with independent logistic classifiers, the system is much better equipped to model data where multiple "
        "classes can apply to the same box."
    ),
    (
        "Darknet-53 acts as a hybrid between the Darknet-19 network used in YOLOv2 and modern residual networks. It still "
        "uses successive 3x3 and 1x1 convolutional layers, but it is significantly larger (53 layers) and now incorporates "
        "shortcut connections. Compared to competing models, Darknet-53 performs on par with state-of-the-art classifiers "
        "but with far more efficiency. It achieves better accuracy than ResNet-101 while being 1.5 times faster, and it "
        "matches the performance of ResNet-152 while being 2 times faster. The developers note that Darknet-53 better "
        "utilizes the GPU and achieves the highest measured floating-point operations per second, largely because ResNets "
        "possess 'way too many layers and aren't very efficient'."
    ),
    (
        "YOLOv3 attempts to solve the small object problem by predicting boxes at three different scales, utilizing a "
        "concept similar to feature pyramid networks. The network takes a feature map, upsamples it by 2x, and concatenates "
        "it with a finer-grained feature map from earlier in the network. This allows the network to combine high-level "
        "semantic information with fine-grained details. Because of these new multi-scale predictions, YOLOv3 successfully "
        "reversed its historical weakness and now achieves relatively high performance on small objects. However, the "
        "unexpected trade-off is that YOLOv3 now exhibits comparatively worse performance when detecting medium and larger-sized objects."
    ),
    (
        "The paper notes that directly estimating coordinate values using MSE is problematic because it treats the points of the "
        "bounding box as independent variables, which fails to consider the integrity of the object itself. Additionally, when "
        "traditional methods calculate the l1 or l2 loss of a box, the loss inherently increases with the scale of the object. "
        "To process this better, researchers shifted to Intersection over Union (IoU) losses, which are scale-invariant. "
        "The YOLOv4 authors utilize CIoU loss because it simultaneously considers the overlapping area, the distance between the "
        "center points of the predicted and ground truth boxes, and the aspect ratio, leading to better convergence speed and "
        "accuracy on the bounding box regression problem."
    ),
    (
        "The developers removed these specific activation functions from their candidate list for practical training reasons. "
        "Specifically, PReLU and SELU were excluded because they are more difficult to train. ReLU6 was excluded because it "
        "is an activation function specially designed for quantization networks, which did not align with their goals. "
        "Instead, they evaluated others like LReLU, Swish, and Mish, ultimately selecting Mish for their 'Bag of Specials'."
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. RUN AGENT FOR EACH QUESTION
# ══════════════════════════════════════════════════════════════════════════════

def run_agent_for_eval(question: str) -> tuple[list[str], str]:
    """
    Invoke the LangGraph YOLO agent.
    Returns strings to prevent RAGAS crashes from unparsed Document objects.
    """
    result = agent_app.invoke({
        "question":     question,
        "chat_history": [],
        "retry_count":  0,
        "documents":    [],
        "session_id":   "ragas-eval",
        "sources":      [],
        "source_details": [],
    })

    raw_contexts = result.get("documents", [])
    answer = result.get("final_answer", "")

    # Safety check: Force cast retrieved context chunks to pure strings
    contexts = [
        doc.page_content if hasattr(doc, 'page_content') else str(doc) 
        for doc in raw_contexts
    ]

    return contexts, answer


def collect_eval_data():
    """Run the agent and build the dataset using exact v0.2+ schema keys."""
    all_contexts = []
    all_answers  = []

    for i, q in enumerate(EVAL_QUESTIONS):
        print(f"\n{'='*60}")
        print(f"  EVAL Q{i+1}/{len(EVAL_QUESTIONS)}: {q[:80]}...")
        print(f"{'='*60}")

        contexts, answer = run_agent_for_eval(q)
        all_contexts.append(contexts)
        all_answers.append(answer)

        print(f"  >> Retrieved {len(contexts)} context chunks")
        print(f"  >> Answer length: {len(answer)} chars")

    # Using the strict naming convention expected by newer versions of RAGAS
    return {
        "user_input":         EVAL_QUESTIONS,
        "reference":          GROUND_TRUTHS,
        "retrieved_contexts": all_contexts,
        "response":           all_answers,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONFIGURE AZURE EVALUATOR MODELS
# ══════════════════════════════════════════════════════════════════════════════

class _FixedTempAzureChatOpenAI(AzureChatOpenAI):
    """
    o4-mini ONLY supports temperature=1.
    RAGAS internally sets temperature=0.01 (faithfulness, context metrics)
    and temperature=0.3 (answer_relevancy), which causes 400 BadRequestError.
    This wrapper silently strips any temperature override before it hits Azure.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("temperature", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("temperature", None)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind(self, **kwargs):
        kwargs.pop("temperature", None)
        return super().bind(**kwargs)

    @property
    def _default_params(self):
        params = super()._default_params
        params.pop("temperature", None)
        return params

    @property
    def _invocation_params(self):
        params = super()._invocation_params
        params.pop("temperature", None)
        return params


def get_ragas_llm():
    """AzureChatOpenAI with temperature override protection for o4-mini."""
    full_url = settings.AZURE_GPT_4O_MINI_ENDPOINT
    base_endpoint = full_url.split("/openai/")[0]

    return _FixedTempAzureChatOpenAI(
        azure_endpoint=base_endpoint,
        azure_deployment=settings.LLM_DEPLOYMENT,
        api_key=settings.AZURE_GPT_4O_MINI_KEY,
        api_version=settings.LLM_API_VERSION,
    )


def get_ragas_embeddings():
    """AzureOpenAIEmbeddings using your text-embedding-3-small deployment."""
    base_endpoint = settings.AZURE_TEXT_EMBEDDING_ENDPOINT.split("/openai/")[0]
    
    return AzureOpenAIEmbeddings(
        azure_endpoint=base_endpoint,
        azure_deployment="text-embedding-3-small",
        api_key=settings.AZURE_TEXT_EMBEDDING_KEY,
        api_version="2023-05-15",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. RUN RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def main():

    print("\n[STEP 1/3] Running agent on evaluation questions...\n")
    eval_data = collect_eval_data()

    print("\n[STEP 2/3] Building dataset...\n")
    dataset = Dataset.from_dict(eval_data)
    print(dataset)

    print("\n[STEP 3/3] Running RAGAS evaluation...\n")
    judge_llm        = get_ragas_llm()
    judge_embeddings = get_ragas_embeddings()

    # Throttling config to prevent Azure 429 Too Many Requests errors
    # max_workers=2 limits concurrency. Increase if your Azure quota allows.
    safe_config = RunConfig(
        timeout=120, 
        max_retries=10, 
        max_wait=60, 
        max_workers=2
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=safe_config,
    )

    # ── Display results ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 70)

    results_df = results.to_pandas()
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    existing_metric_cols = [c for c in metric_cols if c in results_df.columns]

    # Aggregated scores
    aggregated = {}
    print("\n[SCORES] Aggregated Scores:")
    for col in existing_metric_cols:
        mean_val = results_df[col].mean()
        if pd.notna(mean_val):
            aggregated[col] = round(float(mean_val), 4)
            print(f"   {col:<25}: {mean_val:.4f}")

    print("\nPer-Question Breakdown:\n")
    display_cols = ["user_input"] + existing_metric_cols
    display_cols = [c for c in display_cols if c in results_df.columns]
    
    # Truncate user input for terminal readability
    display_df = results_df[display_cols].copy()
    if "user_input" in display_df.columns:
        display_df["user_input"] = display_df["user_input"].apply(lambda x: x[:40] + "..." if len(x) > 40 else x)
    
    print(display_df.to_string(index=False))

    # ── Save outputs ──────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)

    csv_path = "data/ragas_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] CSV saved to: {csv_path}")

    json_path = "data/ragas_results.json"
    per_question_results = []
    
    for i in range(len(EVAL_QUESTIONS)):
        entry = {
            "user_input":         eval_data["user_input"][i],
            "reference":          eval_data["reference"][i],
            "retrieved_contexts": eval_data["retrieved_contexts"][i],
            "response":           eval_data["response"][i],
            "scores": {},
        }
        for col in existing_metric_cols:
            if col in results_df.columns:
                val = results_df.iloc[i][col]
                entry["scores"][col] = round(float(val), 4) if pd.notna(val) else None
        per_question_results.append(entry)

    json_output = {
        "aggregated_scores": aggregated,
        "num_questions":     len(EVAL_QUESTIONS),
        "per_question":      per_question_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] JSON saved to: {json_path}")

    return results_df


if __name__ == "__main__":
    main()