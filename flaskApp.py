from Bani.Bani import Bani
from Bani.core.FAQ import FAQ

import pickle
from flask import Flask, request, jsonify
import os
import logging

logging.basicConfig(level=logging.ERROR)


adoptionFAQ = FAQ(name="Adoption")
babybonusFAQ = FAQ(name="Baby Bonus")
comcareFAQ = FAQ(name="ComCare")
covid19FAQ = FAQ(name="Covid 19")


adoptionFAQ.load("./faqStore")
babybonusFAQ.load("./faqStore")
comcareFAQ.load("./faqStore")
covid19FAQ.load("./faqStore")

masterBot = Bani(
    FAQs=[adoptionFAQ, babybonusFAQ, comcareFAQ, covid19FAQ], modelPath="./latest_model"
)


app = Flask(__name__)


@app.route("/")
def main():
    return jsonify(result="hello world")


@app.route("/answer", methods=["GET"])
def getAnswer():
    global masterConfig
    global interfaces
    params = request.args
    if "question" not in params:
        return "Question not found.", 400
    question = params["question"]

    outputs = masterBot.findClosest(question, K=1)

    code = -1
    answer = None
    similarQns = []

    if outputs[0].maxScore < 0.5:
        # if confidence level not high enough
        # get closest question from each FAQ
        code = 1
        answer = ""
        for out in outputs:
          similarQns.append(out.question.text)

    elif outputs[0].maxScore - outputs[1].maxScore < 0.05:
        code = 2
        answer = outputs[0].answer.text
        similarQns.append(outputs[0].question.text)
        for out in outputs[1:]:
            if outputs[0].maxScore - out.maxScore < 0.05:
                similarQns.append(out.question.text)
            else:
                break

    code = 0 if code==-1 else code
    answer = outputs[0].answer.text.strip() if answer==None else answer
    similarQns = outputs[0].similarQuestions if len(similarQns)==0 else similarQns
    return jsonify(code=code, result=answer, similarQuestions=similarQns)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", 1995), threaded=True)
