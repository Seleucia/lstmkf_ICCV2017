def summary_write(logging_ops,seq_ls_internal,nlayer):
    index=0
    summ_lst=['i','j','f','c','m']
    for item in seq_ls_internal:
      if index%25==0:
          for s in summ_lst:
            for l in range(nlayer):
                logging_ops.histogram_summary(str(index)+'/'+s+'/'+str(nlayer), item[l][s])
                index+=1

