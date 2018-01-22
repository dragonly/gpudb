/*
 * Copyright (c) 2017 Yilong Li <liyilongko@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

int mpsclient_init();
void mpsclient_destroy();
cudaError_t mpsclient_cudaMalloc(void **devPtr, size_t size, uint32_t flags);
cudaError_t mpsclient_cudaFree(void *devPtr);
cudaError_t mpsclient_cudaMemcpy(void *dst, const void *src, size_t size, enum cudaMemcpyKind kind);
cudaError_t mpsclient_cudaMemset(void *devPtr, int32_t value, size_t count);
cudaError_t mpsclient_cudaAdvise(int iarg, int advice);
cudaError_t mpsclient_cudaSetFunction(int index);
cudaError_t mpsclient_cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
cudaError_t mpsclient_cudaSetupArgument(const void *arg, size_t size, size_t offset);
cudaError_t mpsclient_cudaLaunch(const void *func);
cudaError_t mpsclient_cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);
int send_large_buf(int socket, unsigned char *buf, uint32_t size);

