
#pragma once

#include "xop/xop.h"
#include "nccl.h"

namespace xop {
 
#define NCCL_CHECK(cmd)                                                                                           \
do {                                                                                                          \
    ncclResult_t result = cmd;                                                                                \
    if (result != ncclSuccess) {                                                                              \
        printf("[ERROR] NCCL error %s:%d '%s' : %s\n", __FILE__, __LINE__, #cmd, ncclGetErrorString(result)); \
        exit(-1);                                                                                             \
    }                                                                                                         \
} while (0)

class NcclManager {
public:
  void Init(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id){

    this->my_rank = tp_rank;
    this->my_size = tp_size;

    ncclUniqueId tp_uid;
    memcpy(tp_uid.internal, &tp_id[0], NCCL_UNIQUE_ID_BYTES);

    if (this->my_size == 1) {
        this->comm = nullptr;
        return;
    }
    NCCL_CHECK(ncclCommInitRank(&this->comm, this->my_size, tp_uid, this->my_rank));
  }

  std::vector<int64_t> generate_nccl_id() {
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    std::vector<int64_t> ret;
    ret.resize(NCCL_UNIQUE_ID_BYTES / sizeof(int64_t));
    memcpy(ret.data(), nccl_id.internal, NCCL_UNIQUE_ID_BYTES);
    return ret;
  }
  
private:
  
};

}  // namespace xop
