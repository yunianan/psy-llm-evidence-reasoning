"""Optional OpenAI-compatible adapter; only ICD10_* configuration is used."""
import json
import os
from urllib.request import Request, urlopen
from .engine import question_prompt


class APIJudge:
    def __init__(self, *, api_key=None, base_url=None, model=None):
        self.key = (os.environ.get('ICD10_API_KEY', '') if api_key is None else api_key).strip()
        self.base = (os.environ.get('ICD10_BASE_URL', '') if base_url is None else base_url).strip().rstrip('/')
        self.model = (os.environ.get('ICD10_MODEL', '') if model is None else model).strip()
        if not all((self.key,self.base,self.model)): raise ValueError('Set ICD10_API_KEY, ICD10_BASE_URL, ICD10_MODEL')
        self.usage=[]

    def __call__(self,node,messages,*,clinical_profile=None):
        payload={'model':self.model,'messages':question_prompt(node,messages,clinical_profile=clinical_profile),'temperature':0}
        request=Request(self.base+'/chat/completions',data=json.dumps(payload).encode(),
                        headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        try:
            with urlopen(request,timeout=60) as response:raw=json.load(response)
        except Exception as exc:
            raise RuntimeError('ICD10 model request failed: '+type(exc).__name__) from None
        self.usage.append({'node_id':node['id'],'model':raw.get('model'),'usage':raw.get('usage'),
                           'request_id':raw.get('id'),'reference_cost':None,'actual_bill':None})
        choice=raw['choices'][0]
        if choice.get('finish_reason')!='stop': raise ValueError('Incomplete model response')
        judgment=json.loads(choice['message']['content'])
        if node.get('assessment_kind') == 'episode_grouping':
            return judgment  # Session checks the partition and its verbatim evidence.
        if not isinstance(judgment,dict) or 'confidence' not in judgment or (
            judgment.get('answer') in {'yes','no'} and judgment['confidence'] is None
        ):
            raise ValueError('ICD10 model response requires criterion evidence support')
        return judgment


class ManualJudge:
    """Console collection plus human evidence binding, without an API call."""
    def __call__(self,node,messages):
        print('\n'+node['id']+' '+node['question'])
        print('ICD-10: '+node['original_clause'])
        print('source: '+json.dumps(node['source'],ensure_ascii=False))
        if node.get('assessment_kind') == 'episode_grouping' or 'episode_id' in node:
            for message in messages:
                print(message['message_id']+': '+message['text'])
        if node.get('assessment_kind') == 'episode_grouping':
            return json.loads(input('按发作分组 JSON（episodes: [{evidence: [{message_id, quote}]}], complete: true/false）: '))
        answer=input('判断 yes / no / unknown: ').strip()
        evidence=[]
        if answer in {'yes','no'}:
            print('可引用记录：'+', '.join(m['message_id'] for m in messages))
            mid=input('message_id: ').strip(); quote=input('逐字引文: ').strip()
            evidence=[{'message_id':mid,'quote':quote}]
        return {'answer':answer,'rationale':input('判断理由: '),'evidence':evidence,
                'follow_up':input('具体追问: ') if answer=='unknown' else ''}
