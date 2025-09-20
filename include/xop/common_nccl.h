
#pragma once

#include "xop/xop.h"
#include "nccl.h"

namespace xop {
  
struct threadArgs {
  size_t nbytes;
  size_t minbytes;
  size_t maxbytes;
  size_t stepbytes;
  size_t stepfactor;

  int totalProcs;
  int nProcs;
  int proc;
  int nThreads;
  int thread;
  int nGpus;
  int* gpus;
  int localRank;
  void** sendbuffs;
  size_t sendBytes;
  size_t sendInplaceOffset;
  void** recvbuffs;
  size_t recvInplaceOffset;
  ncclUniqueId ncclId;
  ncclComm_t* comms;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  ncclDevComm* devComms;
#endif
  cudaStream_t* streams;

  void** expected;
  size_t expectedBytes;
  int* errors;
  double* bw;
  int* bw_count;

  int reportErrors;

  struct testColl* collTest;

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
  void** sendRegHandles;
  void** recvRegHandles;
#endif
};

}  // namespace xop
