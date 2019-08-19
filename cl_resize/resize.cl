#define OUT_W 320
#define OUT_H 320
#define CH 3

__attribute__((num_simd_work_items(4)))
__attribute__((num_compute_units(2)))
__attribute__((reqd_work_group_size(32,32,1)))
__kernel void resize(int in_w, int in_h, int out_w, int out_h, int offset_in, int offset_out, int dx, int dy, float ratio, __global float * restrict in, __global float * restrict out){
  int row = get_global_id(0);
  int col = get_global_id(1);

  int idx_out = row * OUT_W + col;

  float in1, in2, in3;
  if(row < dy || row >= OUT_H - dy || col < dx || col >= OUT_W - dx){
    in1 = 0.5f;
    in2 = 0.5f;
    in3 = 0.5f;
  }else{
    int idx_in = (int)((float)(row-dy) / ratio) * in_w + (int)((float)(col-dx) / ratio);

    in1 = in[idx_in              ];
    in2 = in[idx_in + offset_in  ];
    in3 = in[idx_in + offset_in*2];
  }

  out[idx_out               ] = in1;
  out[idx_out + offset_out  ] = in2;
  out[idx_out + offset_out*2] = in3;
}
