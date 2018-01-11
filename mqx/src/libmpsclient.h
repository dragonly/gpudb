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
cudaError_t mpsclient_cudaMemcpyHtoD(void *dst, const void *src, size_t size);
cudaError_t mpsclient_cudaMemcpyDtoH(void *dst, const void *src, size_t size);
cudaError_t mpsclient_cudaMemcpyDtoD(void *dst, const void *src, size_t size);
cudaError_t mpsclient_cudaMemcpyDefault(void *dst, const void *src, size_t size);

