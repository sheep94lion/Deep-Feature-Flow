#ifndef MXNET_OPERATOR_CONTRIB_RPN_INV_NORMALIZE_OP_INL_H_
#define MXNET_OPERATOR_CONTRIB_RPN_INV_NORMALIZE_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
    namespace op {
        struct RpnInvNormalizeParam: public dmlc::Parameter<RpnInvNormalizeParam> {
            int num_anchors;
            float *bbox_mean, *bbox_std;
            DMLC_DECLARE_PARAMETER(RpnInvNormalizeParam) {
                DMLC_DECLARE_FIELD(num_anchors)
                    .describe("Num of anchors.");
                DMLC_DECLARE_FIELD(bbox_mean)
                    .describe("Bbox mean.");
                DMLC_DECLARE_FIELD(bbox_std)
                    .describe("Bbox std.");
            }
        };

        inline bool RpnInvNormalizeOpShape(const nnvm::NodeAttrs& attrs,
                                           mxnet::ShapeVector* in_attrs,
                                           mxnet::ShapeVector* out_attrs) {
            return true;
        }

        inline bool RpnInvNormalizeOpType(const nnvm::NodeAttrs& attrs,
                                          std::vector<int>* in_attrs,
                                          std::vector<int>* out_attrs) {
            return true;
        }

        template<typename xpu>
        void RpnInvNormalizeOpForward(const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const std::vector<TBlob>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<TBlob>& outputs) {
            mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
            const TBlob& in_data = inputs[0];
            const TBlob& outdata = outputs[0];
            const RpnInvNormalizeParam& param = nnvm::get<RpnInvNormalizeParam>(attrs.parsed);
            using namespace mxnet_op;
            MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
                MXNET_ASSIGN_REQ_SWITCH(req[0], req_type,  {
                    
                });
            });
        }
    }
}

#endif