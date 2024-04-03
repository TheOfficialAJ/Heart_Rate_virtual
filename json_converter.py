import json
def json_converter(sig:list,time:list):
    l=len(sig)
    json_dict={'sig_n_time':[(sig[i],time[i]) for i in range(l)]}
    json_data=json.dumps(json_dict)
    return json_data

if __name__=='__main__':
    print(json_converter([5,4,3,2,1],[1,2,3,4,5]))#for testing
