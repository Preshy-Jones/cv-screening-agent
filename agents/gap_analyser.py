async def gap_analyser_agent(state: AgentState) -> AgentState:
    required_skills = state.jd_analysis["required_skills"]
    cv_skills = state.cv_analysis["skills"]
    cv_text = state.cv_text
    
    # Embed all required skills at once — one API call
    skill_embeddings = client.embeddings.create(
        input=required_skills,
        model="text-embedding-3-small"
    ).data
    
    # Embed all CV skills at once — one API call
    cv_skill_embeddings = client.embeddings.create(
        input=cv_skills,
        model="text-embedding-3-small"
    ).data
    
    gaps = []
    matches = []
    
    for i, required_skill in enumerate(required_skills):
        req_vector = skill_embeddings[i].embedding
        
        # Check similarity against every CV skill
        best_match = None
        best_score = 0
        
        for j, cv_skill in enumerate(cv_skills):
            cv_vector = cv_skill_embeddings[j].embedding
            score = cosine_similarity(req_vector, cv_vector)
            
            if score > best_score:
                best_score = score
                best_match = cv_skill
        
        if best_score > 0.85:
            matches.append({
                "required": required_skill,
                "matched_to": best_match,
                "confidence": best_score
            })
        elif best_score > 0.70:
            # Possible match — check against full CV text too
            # Maybe it's described differently in a bullet point
            cv_text_embedding = client.embeddings.create(
                input=cv_text[:3000],
                model="text-embedding-3-small"
            ).data[0].embedding
            
            text_score = cosine_similarity(req_vector, cv_text_embedding)
            
            if text_score > 0.75:
                matches.append({
                    "required": required_skill,
                    "matched_to": "implicit in CV text",
                    "confidence": text_score
                })
            else:
                gaps.append(required_skill)
        else:
            gaps.append(required_skill)
    
    match_score = int(len(matches) / len(required_skills) * 100)
    
    state.gap_analysis = {
        "strong_matches": matches,
        "gaps": gaps,
        "match_score": match_score,
        "recommendation": (
            "STRONG_APPLY" if match_score >= 85 else
            "APPLY" if match_score >= 65 else
            "STRETCH" if match_score >= 45 else
            "SKIP"
        )
    }
    
    return state