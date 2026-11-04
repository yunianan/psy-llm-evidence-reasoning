"""Locally authored operational questions; ICD-10 original clauses are kept separate.

Only the clause attached to the current question is sent to a model. Aggregation
uses three-valued expressions and is never delegated to a language model.
"""
import json


def R(n): return {'ref': n}
def ALL(*x): return {'all': list(x)}
def ANY(*x): return {'any': list(x)}
def NOT(x): return {'not': x}
def K(k, xs): return {'at_least': k, 'of': xs}

class Author:
    def __init__(self):
        self.nodes, self.entries, self.targets = {}, [], []

    def node(self, key, question, page, section, clause, original, *, document='WHO_DCR10_1993', pdf_page=None, clinical=False):
        if key in self.nodes: raise ValueError(key)
        self.nodes[key] = {'id': key, 'question': question, 'original_clause': original,
            'assessment_kind': 'differential_or_complex' if clinical else 'symptom',
            'source': {'document_id': document, 'printed_page': page, 'pdf_page': pdf_page or page + 14,
                       'section': section, 'clause': clause,
                       'pdf': 'sources/WHO_DCR10_1993.pdf' if document=='WHO_DCR10_1993' else 'sources/WHO_CDDG_1992.pdf',
                       'text_status': 'human_transcription_or_text_extraction; local_question_is_a_paraphrase'}}
        return R(key)

    def entry(self, key, name, question, page, section, original):
        screen = self.node('entry.'+key, question, page, section, 'entry screen derived from symptom clause; not a diagnostic criterion', original)
        self.entries.append({'id': key, 'label': name, 'screen': screen, 'tree_id': key})

    def target(self, tree, code, label, expression, *, status='criterion_supported_candidate', note=''):
        self.targets.append({'id': tree+':'+code, 'tree_id': tree, 'code': code, 'label': label,
                             'when': expression, 'output_status': status, 'note': note})

    def exclude(self, tree, key, question, page, section, clause, original):
        return self.node(tree+'.'+key, question, page, section, clause, original, clinical=True)


ANXIETY = [
 ('心悸、心跳沉重或心率加快', 'palpitations or pounding heart, or accelerated heart rate'),
 ('出汗', 'sweating'), ('发抖或颤动', 'trembling or shaking'),
 ('口干（不是药物或脱水所致）', 'dry mouth (not due to medication or dehydration)'),
 ('呼吸困难', 'difficulty in breathing'), ('窒息感', 'feeling of choking'),
 ('胸痛或胸部不适', 'chest pain or discomfort'), ('恶心或腹部不适', 'nausea or abdominal distress (e.g. churning in stomach)'),
 ('头晕、站立不稳、晕厥感或头重脚轻', 'feeling dizzy, unsteady, faint, or light-headed'),
 ('现实解体或人格解体感', 'feelings that objects are unreal (derealization), or that the self is distant or "not really here" (depersonalization)'),
 ('害怕失控、发疯或昏倒', 'fear of losing control, "going crazy", or passing out'),
 ('害怕死亡', 'fear of dying'), ('潮热或寒战', 'hot flushes or cold chills'), ('麻木或刺痛感', 'numbness or tingling sensations')]

def anxiety_items(a, tree, page, section, context):
    return [a.node(f'{tree}.symptom_{i}', f'{context}，是否有{zh}？', page, section, f'B({i})', en+'.') for i,(zh,en) in enumerate(ANXIETY,1)]


def build_catalog():
    a=Author()
    # Generalized anxiety: the DCR uses FOUR of 22, including an autonomic item.
    a.entry('gad','广泛性焦虑障碍','是否存在持续紧张、担忧或不祥预感，值得进入此路径核对？',95,'F41.1 A',
            'There must have been a period of at least 6 months with prominent tension, worry, and feelings of apprehension about everyday events and problems.')
    dur=a.node('gad.duration','对于日常事件和问题的显著紧张、担忧及不祥预感是否已持续至少6个月？',95,'F41.1','A',
        'There must have been a period of at least 6 months with prominent tension, worry, and feelings of apprehension about everyday events and problems.')
    g=anxiety_items(a,'gad',96,'F41.1','在这一段持续焦虑期间')
    extra=[('肌肉紧张、酸痛','muscle tension or aches and pains'),('坐立不安且无法放松','restlessness and inability to relax'),
           ('感觉绷紧、忐忑或精神紧张','feeling keyed up, on edge, or mentally tense'),('喉头堵塞感或吞咽困难','a sensation of a lump in the throat, or difficulty in swallowing'),
           ('对轻微惊吓反应过度','exaggerated response to minor surprises or being startled'),('因担心而难集中注意或脑中空白','difficulty in concentrating, or mind "going blank", because of worrying or anxiety'),
           ('持续易激惹','persistent irritability'),('因担心而难以入睡','difficulty in getting to sleep because of worrying')]
    for i,(zh,en) in enumerate(extra,15):
        g.append(a.node(f'gad.symptom_{i}',f'在这一段持续焦虑期间，是否有{zh}？',96 if i<20 else 97,'F41.1',f'B({i})',en+'.'))
    gc=a.exclude('gad','differential','根据患者陈述及现有资料，是否不符合惊恐障碍、恐怖性焦虑障碍、强迫性障碍或疑病性障碍的标准？',97,'F41.1','C',
        'The disorder does not meet the criteria for panic disorder (F41.0), phobic anxiety disorders (F40.-), obsessive-compulsive disorder (F42.-), or hypochondriacal disorder (F45.2).')
    gd=a.exclude('gad','cause','根据已收集的病史及相关资料，是否可排除身体疾病、器质性精神障碍或精神活性物质相关障碍导致本次焦虑？',97,'F41.1','D',
        'The anxiety disorder is not due to a physical disorder, such as hyperthyroidism, an organic mental disorder (F00-F09), or a psychoactive substance-related disorder (F10-F19), such as excess consumption of amfetamine-like substances or withdrawal from benzodiazepines.')
    a.target('gad','F41.1','广泛性焦虑障碍',ALL(dur,K(4,g),K(1,g[:4]),gc,gd))

    a.entry('panic','惊恐障碍','是否有反复突然的强烈恐惧或不适发作，值得进一步核对？',94,'F41.0 A',
        'The individual experiences recurrent panic attacks that are not consistently associated with a specific situation or object and that often occur spontaneously (i.e. the episodes are unpredictable).')
    p=anxiety_items(a,'panic',94,'F41.0','在同一次所评估的惊恐发作中')
    # The items on the second page share the same criterion, with exact pages.
    for i in range(5,15): a.nodes[f'panic.symptom_{i}']['source'].update(printed_page=95,pdf_page=109)
    pa=a.node('panic.recurrent','发作是否反复出现、并非总与特定情境或物体相关且常自发发生？',94,'F41.0','A sentence 1',
        'The individual experiences recurrent panic attacks that are not consistently associated with a specific situation or object and that often occur spontaneously (i.e. the episodes are unpredictable).')
    pb=a.node('panic.not_danger','发作是否并非与明显用力、危险或威胁生命的情境相关？',94,'F41.0','A sentence 2',
        'The panic attacks are not associated with marked exertion or with exposure to dangerous or life-threatening situations.')
    pf=[a.node('panic.'+key,q,94,'F41.0','B('+str(i)+')',en) for i,(key,q,en) in enumerate([
        ('episode','是否是一次可区分的强烈恐惧或不适发作？','it is a discrete episode of intense fear or discomfort;'),
        ('abrupt','发作是否突然开始？','it starts abruptly;'),
        ('peak','是否在几分钟内达到高峰，且至少持续数分钟？','it reaches a maximum within a few minutes and lasts at least some minutes;')],1)]
    pc=a.exclude('panic','cause','根据患者陈述及现有资料，是否发作不是身体疾病、器质性精神障碍或其他精神障碍导致的？',95,'F41.0','C',
        'Panic attacks are not due to a physical disorder, organic mental disorder (F00-F09), or other mental disorders such as schizophrenia and related disorders (F20-F29), mood [affective] disorders (F30-F39), or somatoform disorders (F45.-).')
    a.target('panic','F41.0','惊恐障碍',ALL(pa,pb,*pf,K(4,p),K(1,p[:4]),pc),note='未指定可选的第五位中度/重度编码；症状计数必须来自同一次发作。')

    # Social/specific phobias reference F40.0 B. Those 14 items are repeated only
    # as independent questions, with the shared original source preserved.
    for tree,label in [('social','社交恐怖症'),('specific','特定恐怖症')]:
        a.entry(tree,label,'是否存在相应社交情境或特定对象引起的明显恐惧或回避？',92 if tree=='social' else 93,'F40.1 A' if tree=='social' else 'F40.2 A',
                'marked fear of being the focus of attention' if tree=='social' else 'marked fear of a specific object or situation not included in agoraphobia (F40.0) or social phobia (F40.1);')
        section='F40.1' if tree=='social' else 'F40.2'; page=92 if tree=='social' else 93
        if tree=='social':
            fear=a.node(tree+'.fear','社交情境中是否显著害怕成为注意焦点或行为尴尬受辱？',page,section,'A(1)',
                'marked fear of being the focus of attention, or fear of behaving in a way that will be embarrassing or humiliating;')
            avoid=a.node(tree+'.avoid','是否显著回避成为注意焦点或可能尴尬受辱的社交情境？',page,section,'A(2)',
                'marked avoidance of being the focus of attention, or of situations in which there is fear of behaving in an embarrassing or humiliating way.')
        else:
            fear=a.node(tree+'.fear','是否显著害怕某特定对象或情境，且该对象不属于广场或社交恐怖情境？',page,section,'A(1)',
                'marked fear of a specific object or situation not included in agoraphobia (F40.0) or social phobia (F40.1);')
            avoid=a.node(tree+'.avoid','是否显著回避某特定对象或情境，且不属于广场或社交恐怖情境？',page,section,'A(2)',
                'marked avoidance of a specific object or situation not included in agoraphobia (F40.0) or social phobia (F40.1).')
        sx=anxiety_items(a,tree,91,'F40.0','自本障碍开始后，在同一恐惧情境中的同一次反应里')
        for i in (13,14): a.nodes[f'{tree}.symptom_{i}']['source'].update(printed_page=92,pdf_page=106)
        ac=ALL(K(2,sx),K(1,sx[:4]))
        if tree=='social':
            ss=[a.node(f'social.specific_{i}',f'在相关恐惧社交情境中，是否出现过{zh}？',93,'F40.1',f'B({i})',en) for i,(zh,en) in enumerate([
                ('脸红或发抖','blushing or shaking;'),('害怕呕吐','fear of vomiting;'),('尿便急迫或害怕大小便失控','urgency or fear of micturition or defecation.')],1)]
            ac=ALL(ac,K(1,ss))
        distress=a.node(tree+'.distress','症状或回避是否造成显著情绪痛苦？',93,section,'C part 1',
            'Significant emotional distress is caused by the symptoms or by the avoidance')
        insight=a.node(tree+'.insight','患者是否认识到这些恐惧或回避是过度或不合理的？',93,section,'C part 2',
            'the individual recognizes that these are excessive or unreasonable.')
        restricted=a.node(tree+'.restricted','症状是否'+('局限于或主要出现在' if tree=='social' else '局限于')+'恐惧情境或对该情境的设想？',93 if tree=='social' else 94,section,'D',
            'Symptoms are restricted to, or predominate in, the feared situations or contemplation of the feared situations.' if tree=='social' else 'Symptoms are restricted to the feared situation or contemplation of the feared situation.')
        clauses=[ANY(fear,avoid),ac,distress,insight,restricted]
        if tree=='social':
            clauses.append(a.exclude(tree,'cause','根据患者陈述及现有资料，是否这些症状不是妄想、幻觉、其他所列精神障碍或文化信念所致？',93,section,'E',
                'The symptoms listed in criteria A and B are not the result of delusions, hallucinations, or other disorders such as organic mental disorders (F00-F09), schizophrenia and related disorders (F20-F29), mood [affective] disorders (F30-F39), or obsessive-compulsive disorder (F42.-), and are not secondary to cultural beliefs.'))
        a.target(tree,'F40.1' if tree=='social' else 'F40.2',label,ALL(*clauses),note='B 的上位引文见 F40.0 B；同一情境同一次反应计数。')

    build_depression(a)
    build_other(a)
    build_alcohol(a)
    materialize_calculations(a)
    return {'id':'WHO-ICD10-14-v2-dialogue','standard':'WHO ICD-10',
            'evidence_policy':'dialogue_supported_v2','documents': document_manifest(),
            'entries':a.entries,'nodes':a.nodes,'targets':a.targets,
            'comparison':{'status':'not_implemented','candidate_deletion':False},
            'review_status':'source_transcribed_local_operationalization; independent_clinical_review_pending'}


def materialize_calculations(a):
    """Every numeric rule has its own original threshold clause and trace node."""
    memo={}
    def visit(expr,tree):
        if 'ref' in expr: return expr
        if 'at_least' in expr:
            k=expr['at_least']; refs=[x['ref'] for x in expr['of']]
            signature=json.dumps(expr,sort_keys=True)
            if signature in memo: return memo[signature]
            page,section,clause,quote=threshold_source(tree,k,refs)
            key=f'{tree}.calculation_{sum(n.startswith(tree+".calculation_") for n in a.nodes)+1}'
            ref=a.node(key,f'程序判断：本条所列独立条目是否至少{k}项为是？',page,section,clause,quote)
            a.nodes[key]['calculation']=expr
            memo[signature]=ref
            return ref
        op=next(iter(expr))
        return {op:visit(expr[op],tree) if op=='not' else [visit(x,tree) for x in expr[op]]}
    for t in a.targets:
        t['when']=visit(t['when'],t['tree_id'])


def threshold_source(tree,k,refs):
    if tree=='gad':
        return 96,'F41.1','B lead-in','At least four of the symptoms listed below must be present, at least one of which must be from items (1) to (4):'
    if tree=='panic':
        return 94,'F41.0','B(4)','at least four of the symptoms listed below must be present, one of which must be from items (a) to (d):'
    if tree in {'social','specific'}:
        if 'specific_' in refs[0]: return 93,'F40.1','B social additions','together with at least one of the following symptoms:'
        return 91,'F40.0','B cross-reference from '+('F40.1 B' if tree=='social' else 'F40.2 B'), 'At least two symptoms of anxiety in the feared situation must have been present together, on at least one occasion since the onset of the disorder, and one of the symptoms must have been from items (1) to (4) listed below:'
    if tree=='depression':
        if len(refs)==3:
            return (84,'F32.2','B','All three of the symptoms in criterion B, F32.0, must be present.') if k==3 else (82,'F32.0','B lead-in','At least two of the following three symptoms must be present:')
        return (83 if k in (4,6) else 84),'F32.0' if k==4 else ('F32.1' if k==6 else 'F32.2'),'C',{
            4:'An additional symptom or symptoms from the following list should be present, to give a total of at least four:',
            6:'Additional symptoms from F32.0, criterion C, must be present, to give a total of at least six.',
            8:'Additional symptoms from F32.0, criterion C, must be present, to give a total of at least eight.'}[k]
    if tree=='dysthymia': return 88,'F34.1','C lead-in','During at least some of the periods of depression at least three of the following should be present:'
    if tree=='ptsd': return 100,'F43.1','D(2) lead-in','persistent symptoms of increased psychological sensitivity and arousal (not present before exposure to the stressor), shown by any two of the following:'
    if tree=='hyperkinetic':
        g='G1' if k==6 else ('G2' if k==3 else 'G3')
        return (155 if k in (6,3) else 156),'F90',g+' lead-in',{
            6:'At least six of the following symptoms of inattention have persisted for at least 6 months, to a degree that is maladaptive and inconsistent with the developmental level of the child:',
            3:'At least three of the following symptoms of hyperactivity have persisted for at least 6 months, to a degree that is maladaptive and inconsistent with the developmental level of the child:',
            1:'At least one of the following symptoms of impulsivity has persisted for at least 6 months, to a degree that is maladaptive and inconsistent with the developmental level of the child:'}[k]
    if tree=='alcohol':
        if 'dependence' in refs[0]: return 57,'F1x.2','A lead-in','Three or more of the following manifestations should have occurred together for at least 1 month or, if persisting for periods of less than 1 month, should have occurred together repeatedly within a 12-month period:'
        if 'withdrawal' in refs[0]: return 59,'F10.3','B lead-in','Any three of the following signs must be present:'
        return (49 if 'beh' in refs[0] else 50),'F10.0','B lead-in' if 'beh' in refs[0] else 'C lead-in','There must be dysfunctional behaviour, as evidenced by at least one of the following:' if 'beh' in refs[0] else 'At least one of the following signs must be present:'
    raise ValueError((tree,k,refs))

def document_manifest():
    # Hashes identify the reference scans used to check the page numbering.
    return {
        "WHO_DCR10_1993": {
            "publisher": "WHO", "year": 1993, "isbn": "9241544554",
            "pdf": "sources/WHO_DCR10_1993.pdf",
            "publication_url": "https://www.who.int/publications/i/item/9241544554",
            "sha256": "17627e919b3828a5fecc4d832bda504f38a098862ff54883d4bb717d0e89bc8e",
        },
        "WHO_CDDG_1992": {
            "publisher": "WHO", "year": 1992, "isbn": "9241544228",
            "pdf": "sources/WHO_CDDG_1992.pdf",
            "publication_url": "https://www.who.int/publications-detail-redirect/9241544228",
            "sha256": "6a328c75bc129bb0f91f0bed33baf0be90eeb40f955a5c1436e6a9e51409ab3f",
        },
    }


def build_depression(a):
    a.entry('depression','抑郁发作、复发性抑郁障碍','是否存在低落、兴趣丧失、精力下降或既往反复抑郁发作，值得进入此路径核对？',83,'F32.0 B',
            'loss of interest or pleasure in activities that are normally pleasurable; decreased energy or increased fatiguability.')
    dur=a.node('depression.duration','本次抑郁发作是否已持续至少2周？',81,'F32','G1',
        'The depressive episode should last for at least 2 weeks.')
    no_mania=a.exclude('depression','no_mania','根据患者陈述及现有资料，是否一生中均没有达到轻躁狂或躁狂发作标准的症状？',81,'F32','G2',
        "There have been no hypomanic or manic symptoms sufficient to meet the criteria for hypomanic or manic episode (F30.-) at any time in the individual's life.")
    cause=a.exclude('depression','cause','根据患者陈述及现有资料，是否本次发作不能归因于精神活性物质使用或器质性精神障碍？',82,'F32','G3',
        'The episode is not attributable to psychoactive substance use (F10-F19) or to any organic mental disorder (in the sense of F00-F09).')
    core=[a.node('depression.core_'+str(i),q,82 if i==1 else 83,'F32.0',f'B({i})',en) for i,(q,en) in enumerate([
        ('是否存在相对本人明显异常的抑郁心境，在大部分时间几乎每天出现、基本不随环境变化，且持续至少2周？',
         'depressed mood to a degree that is definitely abnormal for the individual, present for most of the day and almost every day, largely uninfluenced by circumstances, and sustained for at least 2 weeks;'),
        ('本次发作期间是否对通常愉快的活动丧失兴趣或愉悦？','loss of interest or pleasure in activities that are normally pleasurable;'),
        ('本次发作期间是否精力下降或更容易疲劳？','decreased energy or increased fatiguability.')],1)]
    a.nodes['depression.core_1']['source']['continued_on_printed_page']=83
    sx=[a.node('depression.additional_'+str(i),'本次发作期间是否有'+zh+'？',83,'F32.0',f'C({i})',en) for i,(zh,en) in enumerate([
        ('自信或自尊丧失','loss of confidence or self-esteem;'),('不合理的自责或过度、不恰当的内疚','unreasonable feelings of self-reproach or excessive and inappropriate guilt;'),
        ('反复死亡或自杀想法，或任何自杀行为','recurrent thoughts of death or suicide, or any suicidal behaviour;'),
        ('思考或集中注意能力下降，如犹豫不决','complaints or evidence of diminished ability to think or concentrate, such as indecisiveness or vacillation;'),
        ('精神运动活动改变、激越或迟滞（主观或客观）','change in psychomotor activity, with agitation or retardation (either subjective or objective);'),
        ('任何类型的睡眠紊乱','sleep disturbance of any type;'),('食欲增加或下降并伴相应体重变化','change in appetite (decrease or increase) with corresponding weight change.')],1)]
    history=a.node('depression.recurrent','是否至少有一次既往符合轻、中或重度抑郁发作、持续至少2周，并与本次间隔至少2个月且间隔中无显著心境症状？',85,'F33','G1',
        'There has been at least one previous episode, mild (F32.0), moderate (F32.1), or severe (F32.2 or F32.3), lasting a minimum of 2 weeks and separated from the current episode by at least 2 months free from any significant mood symptoms.')
    severe=ALL(K(3,core),K(8,core+sx))
    moderate=ALL(K(2,core),K(6,core+sx),NOT(severe))
    mild=ALL(K(2,core),K(4,core+sx),NOT(K(6,core+sx)))
    no_psych=a.node('depression.no_psychosis','是否明确没有幻觉、妄想或抑郁性木僵？',84,'F32.2','D',
        'There must be no hallucinations, delusions, or depressive stupor.')
    no_schiz=a.exclude('depression','no_schiz','根据患者陈述及现有资料，是否不符合精神分裂症或抑郁型分裂情感性障碍？',84,'F32.3','C',
        'The criteria for schizophrenia (F20.0-F20.3) or schizoaffective disorder, depressive type (F25.1) are not met.')
    psych=a.node('depression.psychosis','是否存在符合本条范围的妄想或幻觉（排除本条所列典型分裂症性内容）？',84,'F32.3','D(1)',
        'delusions or hallucinations, other than those listed as typically schizophrenic in criterion G1(1)b, c, and d for F20.0-F20.3 (i.e. delusions other than those that are completely impossible or culturally inappropriate and hallucinations that are not in the third person or giving a running commentary);')
    stupor=a.node('depression.stupor','现有症状及相关资料是否支持存在抑郁性木僵？',84,'F32.3','D(2)','depressive stupor.',clinical=True)
    base=ALL(dur,no_mania,cause)
    for prefix,hist in [('F32',NOT(history)),('F33',history)]:
        name='抑郁发作' if prefix=='F32' else '复发性抑郁障碍，当前发作'
        for suffix,label,exp in [('0','轻度',mild),('1','中度',moderate),('2','重度不伴精神病性症状',ALL(severe,no_psych)),
                                 ('3','重度伴精神病性症状',ALL(severe,no_schiz,ANY(psych,stupor)))]:
            a.target('depression',prefix+'.'+suffix,name+'：'+label,ALL(base,hist,exp),
                     note='按当前发作分级；不自动补充可选的躯体综合征或心境协调性第五位编码。F32 的非复发路由是为互斥输出设置的分类分支。')


def build_other(a):
    a.entry('dysthymia','心境恶劣','是否存在长期持续或反复的低落心境？',88,'F34.1 A',
        'There must be a period of at least 2 years of constant or constantly recurring depressed mood.')
    d=[a.node('dysthymia.'+key,q,88,'F34.1',clause,en) for key,q,clause,en in [
        ('duration','持续或不断反复的抑郁心境是否已至少2年？','A part 1','There must be a period of at least 2 years of constant or constantly recurring depressed mood.'),
        ('interval','期间正常心境是否很少超过数周？','A part 2','Intervening periods of normal mood rarely last for longer than a few weeks'),
        ('no_hypomania','期间是否明确没有轻躁狂发作？','A part 3','there are no episodes of hypomania.'),
        ('episodes','这2年内是否没有或只有极少数个别低落发作足够严重或持久而达到复发性轻度抑郁障碍标准？','B',
         'None, or very few, of the individual episodes of depression within such a 2-year period should be sufficiently severe or long-lasting to meet the criteria for recurrent mild depressive disorder (F33.0).')]]
    dys=[('精力或活动减少','reduced energy or activity'),('失眠','insomnia'),('自信下降或无能感','loss of self-confidence or feelings of inadequacy'),
         ('集中注意困难','difficulty in concentrating'),('经常哭泣','frequent tearfulness'),('丧失性活动及其他愉快活动的兴趣或乐趣','loss of interest in or enjoyment of sex and other pleasurable activities'),
         ('无望或绝望感','feeling of hopelessness or despair'),('觉得不能应付日常责任','a perceived inability to cope with the routine responsibilities of everyday life'),
         ('对未来悲观或反复思考过去','pessimism about the future or brooding over the past'),('社交退缩','social withdrawal'),('言语减少','reduced talkativeness')]
    ds=[a.node(f'dysthymia.s{i}',f'在至少部分低落期间，是否有{zh}？',88 if i<10 else 89,'F34.1',f'C({i})',en+'.') for i,(zh,en) in enumerate(dys,1)]
    a.target('dysthymia','F34.1','心境恶劣',ALL(*d,K(3,ds)))

    a.entry('mixed','混合性焦虑和抑郁障碍','是否同时有焦虑和抑郁症状？',97,'F41.2 referral to CDDG',
        'specific criteria are not given, other than those already in Clinical descriptions and diagnostic guidelines.')
    def m(key,q,en,clause):
        return a.node('mixed.'+key,q,141,'F41.2',clause,en,document='WHO_CDDG_1992',pdf_page=154)
    ma=m('anxiety','是否存在焦虑症状？','symptoms of both anxiety and depression are present','paragraph 1, symptom component')
    md=m('depression','是否存在抑郁症状？','symptoms of both anxiety and depression are present','paragraph 1, symptom component')
    mt=m('subthreshold','根据患者陈述及现有资料，是否焦虑与抑郁各自都不足以独立形成诊断？',
        'neither set of symptoms, considered separately, is sufficiently severe to justify a diagnosis.','paragraph 1, diagnostic boundary')
    a.nodes['mixed.subthreshold']['assessment_kind']='differential_or_complex'
    auto=m('autonomic','是否至少间歇地有自主神经症状，例如颤抖、心悸、口干或胃部翻动感？',
        'Some autonomic symptoms (tremor, palpitations, dry mouth, stomach churning, etc.) must be present, even if only intermittently;','paragraph 1, autonomic requirement')
    stress=m('no_adjustment','评估是否明确这些症状并非紧密关联显著生活变化或应激事件、需改归适应障碍的情形？',
        'If symptoms that fulfil the criteria for this disorder occur in close association with significant life changes or stressful life events, category F43.2, adjustment disorders, should be used.','paragraph 1, F43.2 precedence')
    no_dys=m('no_dysthymia','评估是否已排除持续性焦虑抑郁属于心境恶劣？','Excludes: persistent anxiety depression (dysthymia) (F34.1)','exclusion')
    a.nodes['mixed.no_adjustment']['assessment_kind']=a.nodes['mixed.no_dysthymia']['assessment_kind']='differential_or_complex'
    a.target('mixed','F41.2','混合性焦虑和抑郁障碍',ALL(ma,md,mt,auto,stress,no_dys),note='依据 DCR p.97 转引 CDDG p.141；无自编症状数量或持续时间阈值。')

    a.entry('adjustment','适应障碍','是否存在明确生活应激后出现的情绪或行为变化？',100,'F43.2 A',
        'Onset of symptoms must occur within 1 month of exposure to an identifiable psychosocial stressor, not of an unusual or catastrophic type.')
    ad=[a.node('adjustment.'+key,q,page,'F43.2',cl,en) for key,q,page,cl,en in [
        ('stressor','是否有可辨认的、并非异常或灾难性的心理社会应激源？',100,'A stressor','an identifiable psychosocial stressor, not of an unusual or catastrophic type.'),
        ('onset','症状是否在接触该应激源后1个月内出现？',100,'A onset','Onset of symptoms must occur within 1 month of exposure'),
        ('symptoms','是否有本条所列心境、神经症性/应激相关或品行方面的症状行为变化，且不是妄想或幻觉？',100,'B symptom type',
         'The individual manifests symptoms or behaviour disturbance of the types found in any of the affective disorders (F30-F39) (except for delusions and hallucinations), any disorders in F40-F48 (neurotic, stress-related, and somatoform disorders) and conduct disorders (F91.-)'),
        ('not_other','根据患者陈述及现有资料，是否尚未达到某一个体障碍的完整标准？',100,'B exclusion','the criteria for an individual disorder are not fulfilled.')]]
    a.nodes['adjustment.not_other']['assessment_kind']='differential_or_complex'
    course=a.node('adjustment.course','应激及其后果结束后，症状是否明确没有持续超过6个月？若应激尚未结束或尚未观察满该时段，回答信息不足。',101,'F43.2','C',
        'Except in prolonged depressive reaction (F43.21), the symptoms do not persist for more than 6 months after the cessation of the stress or its consequences. However, this should not prevent a provisional diagnosis being made if this criterion is not yet fulfilled.')
    prolonged=a.node('adjustment.prolonged','是否为应对长期应激情境的轻度抑郁状态，且持续不超过2年？',101,'F43.21','definition',
        'A mild depressive state occurring in response to a prolonged exposure to a stressful situation but of a duration not exceeding 2 years.')
    a.target('adjustment','F43.2','适应障碍',ALL(*ad,ANY(course,prolonged)),note='未输出可选的各第五位亚型；纵向病程尚未可观察时另保留原文允许的暂定状态。')
    # The engine exposes the provisional route without converting unknown C to yes.
    a.targets[-1]['provisional_when']=ALL(*ad)
    a.targets[-1]['provisional_if_unknown']=['adjustment.course','adjustment.prolonged']

    a.entry('ocd','强迫性障碍','是否有反复侵入的思维、意象或重复仪式行为？',97,'F42 A',
        'Either obsessions or compulsions (or both) are present on most days for a period of at least 2 weeks.')
    o=[a.node('ocd.'+key,q,page,'F42',cl,en) for key,q,page,cl,en in [
        ('duration','强迫观念或强迫行为是否至少2周在大多数日子存在？',97,'A',
         'Either obsessions or compulsions (or both) are present on most days for a period of at least 2 weeks.'),
        ('own','患者是否承认这些观念或行为源于自己的头脑，而非外部人员或力量施加？',97,'B(1)',
         'They are acknowledged as originating in the mind of the patient, and are not imposed by outside persons or influences.'),
        ('repetitive','这些观念或行为是否重复且令人不快？',97,'B(2) part 1','They are repetitive and unpleasant'),
        ('insight','是否至少有一项强迫观念或行为被本人承认为过度或不合理？',97,'B(2) part 2',
         'at least one obsession or compulsion that is acknowledged as excessive or unreasonable must be present.'),
        ('resist','患者是否尝试抵抗且至少有一项抵抗失败？长期存在者抵抗可以很少，但仍需按原条核对。',98,'B(3)',
         'The patient tries to resist them (but resistance to very long-standing obsessions or compulsions may be minimal). At least one obsession or compulsion that is unsuccessfully resisted must be present.'),
        ('no_pleasure','强迫思维或行为本身是否不带来愉悦（区分暂时缓解紧张焦虑）？',98,'B(4)',
         'Experiencing the obsessive thought or carrying out the compulsive act is not in itself pleasurable. (This should be distinguished from the temporary relief of tension or anxiety.)'),
        ('distress','是否造成痛苦或干扰社会/个人功能，通常表现为耗费时间？',98,'C',
         "The obsessions or compulsions cause distress or interfere with the patient's social or individual functioning, usually by wasting time.")]]
    oe=a.exclude('ocd','cause','根据患者陈述及现有资料，是否强迫症状不是其他精神障碍（如精神分裂症相关或心境障碍）的结果？',98,'F42','D',
        'The obsessions or compulsions are not the result of other mental disorders, such as schizophrenia and related disorders (F20-F29) or mood [affective] disorders (F30-F39).')
    a.target('ocd','F42','强迫性障碍',ALL(*o,oe),note='输出总类；不凭强迫观念/行为单项数量推定可选亚型。')

    a.entry('insomnia','非器质性失眠','是否有入睡、维持睡眠困难或睡后不恢复的主诉？',114,'F51.0 A',
        'The individual complains of difficulty falling asleep, difficulty maintaining sleep, or non-refreshing sleep.')
    ins=[a.node('insomnia.'+key,q,114,'F51.0',cl,en) for key,q,cl,en in [
        ('complaint','是否主诉难入睡、难维持睡眠或睡后不能恢复？','A',
         'The individual complains of difficulty falling asleep, difficulty maintaining sleep, or non-refreshing sleep.'),
        ('frequency','上述睡眠困难是否每周至少3次？','B frequency','The sleep disturbance occurs at least three times a week'),
        ('duration','上述频率是否持续至少1个月？','B duration','for at least 1 month.'),
        ('impact','睡眠困难是否造成显著痛苦或日常个人功能受干扰？','C',
         'The sleep disturbance results in marked personal distress or interference with personal functioning in daily living.')]]
    ie=a.exclude('insomnia','cause','根据患者陈述及现有资料，是否没有已知的神经/其他身体疾病、精神活性物质使用障碍或药物等致病性因素？',114,'F51.0','D',
        'There is no known causative organic factor, such as a neurological or other medical condition, psychoactive substance use disorder, or a medication.')
    a.target('insomnia','F51.0','非器质性失眠',ALL(*ins,ie))

    a.entry('ptsd','创伤后应激障碍','是否有异常威胁性或灾难性事件之后出现的持续困扰？',99,'F43.1 A',
        'The patient must have been exposed to a stressful event or situation (either short- or long-lasting) of exceptionally threatening or catastrophic nature, which would be likely to cause pervasive distress in almost anyone.')
    pe=[a.node('ptsd.'+key,q,page,'F43.1',cl,en) for key,q,page,cl,en in [
        ('exposure','是否曾暴露于异常威胁性或灾难性的事件/情境，足以使几乎任何人广泛痛苦？',99,'A',
         'The patient must have been exposed to a stressful event or situation (either short- or long-lasting) of exceptionally threatening or catastrophic nature, which would be likely to cause pervasive distress in almost anyone.'),
        ('reliving','是否持续以侵入闪回、鲜明记忆、反复梦境重新体验，或在相关情境中感到痛苦？',99,'B',
         'There must be persistent remembering or "reliving" of the stressor in intrusive "flashbacks", vivid memories, or recurring dreams, or in experiencing distress when exposed to circumstances resembling or associated with the stressor.'),
        ('avoidance','是否实际或倾向回避类似/相关情境，且这种回避在事件前不存在？',100,'C',
         'The patient must exhibit an actual or preferred avoidance of circumstances resembling or associated with the stressor, which was not present before exposure to the stressor.'),
        ('onset','重现、回避和D项要求是否均在事件发生或一段应激结束后6个月内达到？',100,'E',
         'Criteria B, C, and D must all be met within 6 months of the stressful event or of the end of a period of stress.')]]
    a.nodes['ptsd.reliving']['source']['continued_on_printed_page']=100
    amnesia=a.node('ptsd.amnesia','是否部分或完全不能回忆暴露期间某些重要方面？',100,'F43.1','D(1)',
        'inability to recall, either partially or completely, some important aspects of the period of exposure to the stressor;')
    ar=[a.node(f'ptsd.arousal_{i}',f'事件后是否持续出现事前没有的{zh}？',100,'F43.1',f'D(2)({chr(96+i)})',
        'persistent symptoms of increased psychological sensitivity and arousal (not present before exposure to the stressor): '+en) for i,(zh,en) in enumerate([
        ('入睡或维持睡眠困难','difficulty in falling or staying asleep;'),('易激惹或愤怒爆发','irritability or outbursts of anger;'),
        ('集中注意困难','difficulty in concentrating;'),('警觉过度','hypervigilance;'),('惊跳反应过强','exaggerated startle response.')],1)]
    a.target('ptsd','F43.1','创伤后应激障碍',ALL(*pe,ANY(amnesia,K(2,ar))),note='采用 E 默认的6个月内满足条件；未启用原文允许另行明示的延迟起病研究扩展。')

    build_hyperkinetic(a)
    build_delusional(a)


def build_hyperkinetic(a):
    a.entry('hyperkinetic','多动性障碍（ADHD工作入口）','是否存在注意困难、过度活动或冲动的持续表现？',155,'F90 note',
        'The research diagnosis of hyperkinetic disorder requires the definite presence of abnormal levels of inattention, hyperactivity, and restlessness that are pervasive across situations and persistent over time')
    groups=[('inattention',155,'G1',6,[
        ('不注意细节或粗心犯错','often fails to give close attention to details, or makes careless errors in schoolwork, work, or other activities'),
        ('难维持任务或游戏注意','often fails to sustain attention in tasks or play activities'),
        ('似乎不听别人讲话','often appears not to listen to what is being said to him or her'),
        ('不能遵从指令或完成作业家务工作，且不是因对立或不理解指令','often fails to follow through on instructions or to finish schoolwork, chores, or duties in the workplace (not because of oppositional behaviour or failure to understand instructions)'),
        ('组织任务或活动困难','is often impaired in organizing tasks and activities'),
        ('回避或强烈不喜欢需持续用脑的任务','often avoids or strongly dislikes tasks, such as homework, that require sustained mental effort'),
        ('丢失任务活动所需物品','often loses things necessary for certain tasks or activities, such as school assignments, pencils, books, toys, or tools'),
        ('易被外界刺激分心','is often easily distracted by external stimuli'),('日常健忘','is often forgetful in the course of daily activities')]),
        ('hyperactivity',155,'G2',3,[
        ('手脚不停或坐着扭动','often fidgets with hands or feet or squirms on seat'),
        ('在应保持坐位的课堂或其他情境离座','leaves seat in classroom or in other situations in which remaining seated is expected'),
        ('在不适当场合过度跑爬；青少年或成人可仅有坐立不安感','often runs about or climbs excessively in situations in which it is inappropriate (in adolescents or adults, only feelings of restlessness may be present)'),
        ('游戏过于吵闹或难以安静休闲','is often unduly noisy in playing or has difficulty engaging quietly in leisure activities'),
        ('持续过度运动，基本不随社交情境或要求改变','exhibits a persistent pattern of excessive motor activity that is not substantially modified by social context or demands')]),
        ('impulsivity',156,'G3',1,[
        ('问题没问完就抢答','often blurts out answers before questions have been completed'),
        ('不能排队或等待游戏/群体活动轮次','often fails to wait in lines or await turns in games or group situations'),
        ('打断或侵扰别人','often interrupts or intrudes on others (e.g. butts into others\' conversations or games)'),
        ('话多且不能适当回应社交限制','often talks excessively without appropriate response to social constraints')])]
    gates=[]
    for group,page,clause,k,items in groups:
        qualifier=a.node(f'hyperkinetic.{group}_duration',f'{clause}这组症状是否持续至少6个月，达到不适应且与发育水平不一致的程度？',page,'F90',clause+' lead-in',
            'have persisted for at least 6 months, to a degree that is maladaptive and inconsistent with the developmental level of the child' if group!='impulsivity' else
            'has persisted for at least 6 months, to a degree that is maladaptive and inconsistent with the developmental level of the child')
        xs=[a.node(f'hyperkinetic.{group}_{i}',f'在这段至少6个月的评估时段内，是否{zh}？',156 if group=='hyperactivity' and i>1 else page,'F90',f'{clause}({i})',en+'.') for i,(zh,en) in enumerate(items,1)]
        gates.extend([qualifier,K(k,xs)])
    onset=a.node('hyperkinetic.onset','该障碍的起病是否不晚于7岁？',156,'F90','G4','Onset of the disorder is no later than the age of 7 years.')
    settings=a.node('hyperkinetic.settings','这些标准是否在不止一种情境中满足（例如家与学校），而非仅仅某一场景有个别症状？',156,'F90','G5 criterion',
        'The criteria should be met for more than a single situation, e.g. the combination of inattention and hyperactivity should be present both at home and at school, or at both school and another setting where children are observed, such as a clinic.')
    corroboration=a.node('hyperkinetic.corroboration','跨情境资料是否有适当的多来源支持，而非只凭家长转述课堂行为？不足时需收集教师/记录等资料。',156,'F90','G5 evidential note',
        'Evidence for cross-situationality will ordinarily require information from more than one source; parental reports about classroom behaviour, for instance, are unlikely to be sufficient.')
    impact=a.node('hyperkinetic.impact','G1至G3症状是否造成临床显著痛苦或社会、学业、职业功能受损？',156,'F90','G6',
        'The symptoms in G1-G3 cause clinically significant distress or impairment in social, academic, or occupational functioning.')
    excludes=[]
    for key,label in [('development','广泛性发育障碍（F84）'),('mania','躁狂发作（F30）'),('depression','抑郁发作（F32）'),('anxiety','焦虑障碍（F41）')]:
        excludes.append(a.exclude('hyperkinetic','exclude_'+key,f'根据患者陈述及现有资料，是否本障碍不符合{label}的标准？',156,'F90','G7',
            'The disorder does not meet the criteria for pervasive developmental disorders (F84.-), manic episode (F30.-), depressive episode (F32.-), or anxiety disorders (F41.-).'))
    conduct=a.node('hyperkinetic.conduct','根据患者陈述及现有资料，是否同时符合品行障碍F91的完整标准？',157,'F90.1','definition',
        'The general criteria for both hyperkinetic disorder (F90) and conduct disorders (F91.-) must be met.',clinical=True)
    base=ALL(*gates,onset,settings,corroboration,impact,*excludes)
    a.target('hyperkinetic','F90.0','活动与注意障碍',ALL(base,NOT(conduct)),note='F90.0 同时要求不符合 F91；按具体病史和症状核对，不凭“没有提及”或“未被诊断”判否。')
    a.target('hyperkinetic','F90.1','多动性品行障碍',ALL(base,conduct),note='F91 需依据其完整标准核对；能由对话支持的条目可判断，信息不足则补采。')


def build_delusional(a):
    a.entry('delusional','妄想障碍','是否存在持续的妄想性信念，值得进行专业核对？',70,'F22.0 A',
        'A delusion or a set of related delusions, other than those listed as typically schizophrenic in criterion G1(1)b or d for F20.0-F20.3')
    de=[a.node('delusional.'+key,q,70,'F22.0',cl,en,clinical=clinical) for key,q,cl,en,clinical in [
        ('delusion','根据患者陈述及现有资料，是否存在一项或一组相关妄想，且不属于本条排除的典型分裂症性妄想？','A',
         'A delusion or a set of related delusions, other than those listed as typically schizophrenic in criterion G1(1)b or d for F20.0-F20.3 (i.e. other than completely impossible or culturally inappropriate), must be present.',True),
        ('duration','上述妄想是否已存在至少3个月？','B','The delusion(s) in criterion A must be present for at least 3 months.',False),
        ('no_schiz','根据患者陈述及现有资料，是否不符合精神分裂症F20.0至F20.3的一般标准？','C',
         'The general criteria for schizophrenia (F20.0-F20.3) are not fulfilled.',True),
        ('hallucinations','是否明确没有持续幻觉；若偶有短暂听幻觉，是否也不是第三人称或评论性声音？','D',
         'There must be no persistent hallucinations in any modality (but there may be transitory or occasional auditory hallucinations that are not in the third person or giving a running commentary).',False),
        ('mood','是否有明确证据显示妄想在没有心境紊乱时仍存在（允许间歇抑郁）？','E',
         'Depressive symptoms (or even a depressive episode (F32.-)) may be present intermittently, provided that the delusions persist at times when there is no disturbance of mood.',False),
        ('cause','根据患者陈述及现有资料，是否无F00至F09器质性精神障碍或物质所致精神病性障碍的证据？','F',
         'There must be no evidence of primary or secondary organic mental disorder as listed under F00-F09, or of a psychotic disorder due to psychoactive substance use (F1x.5).',True)]]
    a.target('delusional','F22.0','妄想障碍',ALL(*de))


def build_alcohol(a):
    a.entry('alcohol','酒精使用相关问题','是否存在近期饮酒、持续饮酒相关损害、失控或减停酒后的症状，值得进入酒精路径？',48,'F10',
        'Mental and behavioural disorders due to use of alcohol')
    alcohol=a.node('alcohol.exposure','是否有证据确认所涉及物质为酒精？',48,'F10','substance identification','Mental and behavioural disorders due to use of alcohol')
    # Intoxication: general plus alcohol-specific dysfunctional behaviour/signs.
    intox=a.node('alcohol.intox_dose','是否有明确证据近期饮酒量足以与急性中毒相符？',48,'F1x.0','G1',
        'There must be clear evidence of recent use of a psychoactive substance (or substances) at sufficiently high dose levels to be consistent with intoxication.')
    significant=a.node('alcohol.intox_significance','症状是否符合酒精作用且严重到产生临床重要的意识、认知、知觉、情感或行为变化？',49,'F1x.0','G2',
        'There must be symptoms or signs of intoxication compatible with the known actions of the particular substance (or substances), as specified below, and of sufficient severity to produce disturbances in the level of consciousness, cognition, perception, affect, or behaviour that are of clinical importance.')
    ie=a.exclude('alcohol','intox_cause','根据患者陈述及现有资料，是否症状不能由无关身体疾病解释，也不能由其他精神或行为障碍更好解释？',49,'F1x.0','G3',
        'The symptoms or signs present cannot be accounted for by a medical disorder unrelated to substance use, and not better accounted for by another mental or behavioural disorder.')
    beh=[('脱抑制','disinhibition'),('好争辩','argumentativeness'),('攻击行为','aggression'),('心境不稳定','lability of mood'),('注意受损','impaired attention'),('判断力受损','impaired judgement'),('个人功能受干扰','interference with personal functioning')]
    ib=[a.node(f'alcohol.intox_beh_{i}',f'本次饮酒中毒时是否出现{zh}？',49 if i<4 else 50,'F10.0',f'B({i})',en+'.') for i,(zh,en) in enumerate(beh,1)]
    signs=[('步态不稳','unsteady gait'),('站立困难','difficulty in standing'),('言语含糊','slurred speech'),('眼球震颤','nystagmus'),('意识水平下降','decreased level of consciousness (e.g. sopor, coma)'),('面部潮红','flushed face'),('结膜充血','conjunctival injection')]
    ist=[a.node(f'alcohol.intox_sign_{i}',f'本次饮酒中毒时是否有事实支持{zh}？',50,'F10.0',f'C({i})',en+'.') for i,(zh,en) in enumerate(signs,1)]
    a.target('alcohol','F10.0','酒精急性中毒',ALL(alcohol,intox,significant,ie,K(1,ib),K(1,ist)),note='不自动指定并发症第五位编码。')
    # Dependence: 3 of 6 CO-OCCUR for one month or repeatedly in 12 months.
    dep=[('强烈饮酒欲望或强迫感','a strong desire or sense of compulsion to take the substance'),
         ('控制饮酒开始、结束或饮量能力受损','impaired capacity to control substance-taking behaviour in terms of its onset, termination, or levels of use, as evidenced by: the substance being often taken in larger amounts or over a longer period than intended; or by a persistent desire or unsuccessful efforts to reduce or control substance use'),
         ('减停饮酒发生生理戒断或为缓解/避免戒断而饮酒','a physiological withdrawal state (see F1x.3 and F1x.4) when substance use is reduced or ceased, as evidenced by the characteristic withdrawal syndrome for the substance, or by use of the same (or closely related) substance with the intention of relieving or avoiding withdrawal symptoms'),
         ('耐受增加，即明显增量才能达到效果或同量效果明显下降','evidence of tolerance to the effects of the substance, such that there is a need for significantly increased amounts of the substance to achieve intoxication or the desired effect, or a markedly diminished effect with continued use of the same amount of the substance'),
         ('饮酒占据生活，减少其他兴趣或大量时间用于获得、饮用及恢复','preoccupation with substance use, as manifested by important alternative pleasures or interests being given up or reduced because of substance use; or a great deal of time being spent in activities necessary to obtain, take, or recover from the effects of the substance'),
         ('已知或应知明确危害仍持续饮酒','persistent substance use despite clear evidence of harmful consequences (see F1x.1), as evidenced by continued use when the individual is actually aware, or may be expected to be aware, of the nature and extent of harm')]
    ds=[a.node(f'alcohol.dependence_{i}',f'在所评估的同一依赖时段，是否有{zh}？',57,'F1x.2',f'A({i})',en+'.') for i,(zh,en) in enumerate(dep,1)]
    timing=a.node('alcohol.dependence_timing','达到数量要求的至少3种表现是否共同出现至少1个月，或若每次不足1个月则在12个月内反复共同出现？',57,'F1x.2','A lead-in',
        'Three or more of the following manifestations should have occurred together for at least 1 month or, if persisting for periods of less than 1 month, should have occurred together repeatedly within a 12-month period:')
    a.target('alcohol','F10.2','酒精依赖综合征',ALL(alcohol,K(3,ds),timing),note='不将多个互不重叠时段的症状拼接计数；未指定可选的缓解/使用状态。')
    harm=a.node('alcohol.harm','是否有明确证据饮酒造成或显著促成身体或心理损害？仅有社会不赞同不足。',56,'F1x.1','A',
        'There must be clear evidence that the substance use was responsible for (or substantially contributed to) physical or psychological harm, including impaired judgement or dysfunctional behaviour, which may lead to disability or have adverse consequences for interpersonal relationships.')
    identified=a.node('alcohol.harm_identified','上述损害的性质是否能清楚识别并具体说明？',56,'F1x.1','B','The nature of the harm should be clearly identifiable (and specified).')
    ht=a.node('alcohol.harm_duration','该使用模式是否持续至少1个月或在12个月内反复发生？',57,'F1x.1','C',
        'The pattern of use has persisted for at least 1 month or has occurred repeatedly within a 12-month period.')
    he=a.exclude('alcohol','harm_exclusion','根据同一时段的症状及病史，是否不符合酒精相关的其他精神或行为障碍（急性中毒除外）？',57,'F1x.1','D',
        'The disorder does not meet the criteria for any other mental or behavioural disorder related to the same drug in the same time period (except for acute intoxication, F1x.0).')
    a.target('alcohol','F10.1','酒精有害使用',ALL(alcohol,harm,identified,ht,he))
    cessation=a.node('alcohol.withdrawal_cessation','是否有明确证据在反复、通常长期或大量饮酒后近期停酒或减量？',58,'F1x.3','G1',
        'There must be clear evidence of recent cessation or reduction of substance use after repeated, and usually prolonged and/or high-dose, use of that substance.')
    we=a.exclude('alcohol','withdrawal_cause','根据患者陈述及现有资料，是否这些症状不能由无关身体疾病或其他精神行为障碍更好解释？',58,'F1x.3','G3',
        'Symptoms and signs are not accounted for by a medical disorder unrelated to substance use, and not better accounted for by another mental or behavioural disorder.')
    signs=[('舌、眼睑或伸手时震颤','tremor of the tongue, eyelids, or outstretched hands'),('出汗','sweating'),('恶心、干呕或呕吐','nausea, retching, or vomiting'),
           ('心动过速或高血压','tachycardia or hypertension'),('精神运动性激越','psychomotor agitation'),('头痛','headache'),('失眠','insomnia'),('不适或虚弱','malaise or weakness'),
           ('短暂视、触、听幻觉或错觉','transient visual, tactile, or auditory hallucinations or illusions'),('癫痫大发作','grand mal convulsions')]
    ws=[a.node(f'alcohol.withdrawal_{i}',f'本次减停酒后的同一戒断期是否有{zh}？',59,'F10.3',f'B({i})',en+'.') for i,(zh,en) in enumerate(signs,1)]
    delirium=a.node('alcohol.delirium','根据具体症状、病程及现有资料，是否符合F05谵妄的完整标准？缺少相关资料时回答信息不足。',61,'F1x.4','B','The criteria for delirium (F05.-) must be met.',clinical=True)
    wb=ALL(alcohol,cessation,we,K(3,ws))
    a.target('alcohol','F10.3','酒精戒断状态',ALL(wb,NOT(delirium)))
    a.target('alcohol','F10.4','酒精戒断状态伴谵妄',ALL(wb,delirium))
    # F1x.5 and F1x.6 have direct research clauses; outside-scope references
    # require clinician evidence, rather than invented criteria or absent=no.
    ps=[a.node('alcohol.psychotic_'+key,q,62,'F1x.5',cl,en,clinical=clinical) for key,q,cl,en,clinical in [
        ('symptoms','根据具体描述及现有资料，是否存在符合本条性质的精神病性症状？','A symptom premise','Onset of psychotic symptoms',True),
        ('onset','精神病性症状是否在使用酒精期间或其后2周内起病？','A','Onset of psychotic symptoms must occur during or within 2 weeks of substance use.',False),
        ('duration_min','精神病性症状是否持续超过48小时？','B','The psychotic symptoms must persist for more than 48 hours.',False),
        ('duration_max','该障碍病程是否不超过6个月？','C','Duration of the disorder must not exceed 6 months.',False)]]
    a.target('alcohol','F10.5','酒精所致精神病性障碍',ALL(alcohol,*ps),note='根据具体描述核对症状性质，不能只凭自报诊断名称认定满足。')
    am=[a.node('alcohol.amnesic_'+key,q,page,'F1x.6',cl,en,clinical=clinical) for key,q,page,cl,en,clinical in [
        ('recent','是否有新材料学习受损，严重到干扰日常生活？',62,'A(1)','a defect of recent memory (impaired learning of new material) to a degree sufficient to interfere with daily living;',False),
        ('past','是否回忆既往经历的能力下降？',62,'A(2)','a reduced ability to recall past experiences.',False),
        ('immediate','根据患者陈述及现有资料，是否即刻回忆缺陷不存在或相对不存在？',62,'B(1)','defect in immediate recall (as tested, for example, by the digit span);',True),
        ('consciousness','根据患者陈述及现有资料，是否不存在或相对不存在意识混浊及F05 A所定义注意障碍？',62,'B(2)','clouding of consciousness and disturbance of attention, as defined in F05.-, criterion A;',True),
        ('dementia','根据患者陈述及现有资料，是否不存在或相对不存在总体智力下降/痴呆？',62,'B(3)','global intellectual decline (dementia).',True),
        ('cause','体格、神经检查、实验室或病史是否无可解释这些症状的非物质相关脑病客观证据？',63,'C',
         'There is no objective evidence from physical and neurological examination, laboratory tests, or history of a disorder or disease of the brain (especially involving bilaterally the diencephalic and medial temporal structures), other than that related to substance use, that can reasonably be presumed to be responsible for the clinical manifestations described under criterion A.',True)]]
    a.target('alcohol','F10.6','酒精相关遗忘综合征',ALL(alcohol,*am))
    for i in ('immediate','consciousness','dementia'):
        a.nodes['alcohol.amnesic_'+i]['original_clause']='All of the following are absent (or relatively absent): '+a.nodes['alcohol.amnesic_'+i]['original_clause']
    link=a.node('alcohol.residual_link','本次残留或迟发综合征是否与酒精有明确关系；若停用后起病，是否有强有力的关联证据？',63,'F1x.7','A',
        'Conditions and disorders meeting the criteria for the individual syndromes listed below should be clearly related to substance use. Where onset of the condition or disorder occurs subsequent to use of psychoactive substances, strong evidence should be provided to demonstrate a link.',clinical=True)
    details=a.node('alcohol.residual_description','收集的材料是否已清楚记载残留状态的类型、严重度、病程以及研究需要的完整描述？',63,'F1x.7','Comments',
        'In view of the considerable variation in this category, the characteristics of such residual states or conditions should be clearly documented in terms of their type, severity, and duration. For research purposes full descriptive details should be specified.',clinical=True)
    for code,label,quote in [
        ('F10.70','闪回','Flashbacks'),
        ('F10.71','残留人格或行为障碍','The general criteria for F07.-, personality and behavioural disorder due to brain disease, damage and dysfunction, must be met.'),
        ('F10.72','残留心境障碍','The criteria for F06.3, organic mood [affective] disorder, must be met.'),
        ('F10.73','痴呆','The general criteria for dementia (F00-F03) must be met.'),
        ('F10.74','其他持续性认知损害','The criteria for F06.7, mild cognitive disorder, must be met, except for the exclusion of psychoactive substance use in criterion D.'),
        ('F10.75','迟发精神病性障碍','The general criteria for F1x.5, psychotic disorder, must be met, except with regard to the onset of the disorder, which is more than 2 weeks but not more than 6 weeks after substance use.')]:
        verified=a.node('alcohol.residual_'+code[-2:],f'根据所收集的症状、病程及相关资料，是否属于{label}，并符合本条引用的完整综合征条件？',63,code.replace('10','1x'),
            'syndrome / B if applicable',quote,clinical=True)
        a.target('alcohol',code,'酒精相关'+label,ALL(alcohol,link,details,verified),
            note='交叉引用条件须有具体事实支持；原文明示的客观检查或病程证据缺失时保持未知，不以自述替代检查结果。')
