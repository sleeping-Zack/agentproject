from agent.verifier import AnswerVerifier


def _cleaning_evidence():
    return [
        {
            "id": "6d726f4604d9e7ebcf59a932ae9e9c00:2",
            "content": (
                "滚刷清洁力更强但易缠绕毛发；胶刷不易缠绕但清洁缝隙灰尘效果稍差。"
                "边刷将墙边、角落的灰尘扫向主吸口，提高边缘清洁效果。"
            ),
        },
        {
            "id": "d294281a5769e0a3d1455fbf637ee572:19",
            "content": (
                "清理床底、沙发底的缠绕物，确保机器人可顺利进入，"
                "并在APP中对该区域设置深度拖扫，增大吸力和拖地次数。"
            ),
        },
        {
            "id": "a29a4ba5a539a2a462def8f91eb98ccc:39",
            "content": (
                "定期清理耗材和传感器，及时更换磨损配件，避免机器人磕碰、摔落。"
                "不同品牌配件尺寸、规格不同，非原装配件会影响清洁效果。"
            ),
        },
        {
            "id": "928c24dbdeeb1c0a109f5620c8bac429:40",
            "content": (
                "扫地时风道有异响、吸力不稳定时，检测风道是否有杂物卡顿、"
                "电机是否异常；清理风道杂物，或联系售后检测电机。"
            ),
        },
    ]


def test_systematic_cleaning_diagnostic_is_not_rejected_as_ungrounded():
    evidence = _cleaning_evidence()
    citations = "\n".join(f"[{item['id']}]" for item in evidence)
    answer = (
        "清洁效果下降时，应按以下顺序排查：\n"
        "1. 清理滚刷、胶刷及边刷缠绕物，并检查主吸口和风道是否堵塞；\n"
        "2. 定期更换磨损耗材，并使用原装配件；\n"
        "3. 清理床底、沙发底障碍物，在APP中设置深度拖扫；\n"
        "4. 清洁传感器，避免机器人磕碰；\n"
        "5. 若风道有异响或吸力不稳定，清理杂物或联系售后检测电机。\n\n"
        f"引用来源：\n{citations}"
    )

    result = AnswerVerifier().verify(
        query="扫地机器人最近清洁效果下降，应该如何系统排查？",
        answer=answer,
        evidence=evidence,
        scene="rag",
    )

    assert result.passed is True
    assert result.citation_validity == 1.0
    assert result.citation_coverage == 1.0
    assert result.unsupported_claim_rate <= 0.5


def test_valid_citations_do_not_let_fabricated_specs_through():
    evidence = _cleaning_evidence()
    answer = (
        "该机型配备5200mAh电池。[6d726f4604d9e7ebcf59a932ae9e9c00:2]\n"
        "支持AI视觉识别。[6d726f4604d9e7ebcf59a932ae9e9c00:2]\n"
        "支持自动集尘。[6d726f4604d9e7ebcf59a932ae9e9c00:2]\n"
        "可以在水中工作。[6d726f4604d9e7ebcf59a932ae9e9c00:2]"
    )

    result = AnswerVerifier().verify(
        query="扫地机器人清洁效果下降怎么办",
        answer=answer,
        evidence=evidence,
        scene="rag",
    )

    assert result.passed is False
    assert "unsupported_claim_rate_exceeded" in result.reasons


def test_markdown_emphasis_cannot_hide_a_fabricated_claim():
    evidence = _cleaning_evidence()
    result = AnswerVerifier().verify(
        query="扫地机器人清洁效果下降怎么办",
        answer=(
            "**电池容量为5200mAh。"
            "[6d726f4604d9e7ebcf59a932ae9e9c00:2]**"
        ),
        evidence=evidence,
        scene="rag",
    )

    assert len(result.claim_support) == 1
    assert result.claim_support[0]["supported"] is False
    assert result.passed is False
