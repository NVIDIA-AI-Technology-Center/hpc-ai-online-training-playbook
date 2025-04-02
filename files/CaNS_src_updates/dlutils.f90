! SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: BSD-3-Clause
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! 3. Neither the name of the copyright holder nor the names of its
!    contributors may be used to endorse or promote products derived from
!    this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

module mod_dlutils
  use mod_types
#if defined(_OPENACC)
  use cudafor
#endif
  implicit none

  integer, allocatable :: splits(:)
  integer, allocatable :: sendsizes(:), senddisps(:)
  integer, allocatable :: recvsizes(:), recvdisps(:)
  real(rp), allocatable :: dl_work(:)
#if defined(_OPENACC)
  attributes(device) :: dl_work
#endif

  contains

  subroutine cmpt_wallstresses_2d(n,is_bound,dzci,visc,u,v,p,tauxz,tauyz,tauzz)
    use mod_param, only: cbcpre
    implicit none
    integer , intent(in ), dimension(3) :: n
    logical , intent(in ), dimension(0:1,3) :: is_bound
    real(rp), intent(in ), dimension(0:)    :: dzci
    real(rp), intent(in )                   :: visc
    real(rp), intent(in ), dimension(0:,0:,0:) :: u,v,p
    real(rp), intent(out), dimension(1:,1:,0:) :: tauxz,tauyz,tauzz !(1:nx,1:ny,0:1) with 0:1 -> top:bottom
    !
    integer :: i,j,nx,ny,nz
    !
    !$acc wait
    !
    ! transpose arrays on the fly
    nx = n(1); ny = n(2); nz = n(3)
    if(is_bound(0,3).and.cbcpre(0,3)//cbcpre(1,3) /= 'PP') then
      !$acc parallel loop collapse(2) default(present)
      do j=1,ny
        do i=1,nx
          tauxz(i,j,0) = (u(i,j,1 )-u(i,j,0   ))*dzci(0)*visc
        end do
      end do
    end if
    if(is_bound(1,3).and.cbcpre(0,3)//cbcpre(1,3) /= 'PP') then
      !$acc parallel loop collapse(2) default(present)
      do j=1,ny
        do i=1,nx
          tauxz(i,j,1) = (u(i,j,nz)-u(i,j,nz+1))*dzci(nz)*visc
        end do
      end do
    end if
    !
    if(is_bound(0,3).and.cbcpre(0,3)//cbcpre(1,3) /= 'PP') then
      !$acc parallel loop collapse(2) default(present)
      do j=1,ny
        do i=1,nx
          tauyz(i,j,0) = (v(i,j,1 )-v(i,j,0   ))*dzci(0)*visc
        end do
      end do
    end if
    if(is_bound(1,3).and.cbcpre(0,3)//cbcpre(1,3) /= 'PP') then
      !$acc parallel loop collapse(2) default(present)
      do j=1,ny
        do i=1,nx
          tauyz(i,j,1) = (v(i,j,nz)-v(i,j,nz+1))*dzci(nz)*visc
        end do
      end do
    end if
    !
    if(is_bound(0,3).and.cbcpre(0,3)//cbcpre(1,3) /= 'PP') then
      !$acc parallel loop collapse(2) default(present)
      do j=1,ny
        do i=1,nx
          tauzz(i,j,0) = 0.5*(p(i,j,1 )+p(i,j,0   ))
        end do
      end do
    end if
    if(is_bound(1,3).and.cbcpre(0,3)//cbcpre(1,3) /= 'PP') then
      !$acc parallel loop collapse(2) default(present)
      do j=1,ny
        do i=1,nx
          tauzz(i,j,1) = 0.5*(p(i,j,nz)+p(i,j,nz+1))
        end do
      end do
    end if
  end subroutine cmpt_wallstresses_2d

  subroutine write_sample(pred, label, fname)
    use hdf5
    character(len=*) :: fname
    real(rp), intent(in) :: pred(:,:,:,:), label(:,:,:,:)
    integer(HID_T) :: in_file_id
    integer(HID_T) :: out_file_id
    integer(HID_T) :: dset_id
    integer(HID_T) :: dspace_id

    integer :: err

    block
      integer(HSIZE_T) :: dims(size(shape(pred)))

      call h5open_f(err)
      call h5fcreate_f (fname, H5F_ACC_TRUNC_F, out_file_id, err)

      dims(:) = shape(pred)
      call h5screate_simple_f(size(shape(pred)), dims, dspace_id, err)
      call h5dcreate_f(out_file_id, "pred", H5T_NATIVE_REAL, dspace_id, dset_id, err)
      call h5dwrite_f(dset_id, H5T_NATIVE_REAL, pred, dims, err)
      call h5dclose_f(dset_id, err)
      call h5sclose_f(dspace_id, err)

      dims(:) = shape(label)
      call h5screate_simple_f(size(shape(label)), dims, dspace_id, err)
      call h5dcreate_f(out_file_id, "label", H5T_NATIVE_REAL, dspace_id, dset_id, err)
      call h5dwrite_f(dset_id, H5T_NATIVE_REAL, label, dims, err)
      call h5dclose_f(dset_id, err)
      call h5sclose_f(dspace_id, err)

      call h5fclose_f(out_file_id, err)
      call h5close_f(err)
    end block
  end subroutine

  subroutine reshaped_copy(x, nx, y, ny, lb, ub)
    use mod_param, only: trainbs
    implicit none
    integer :: nx(2), ny(2), lb, ub
    real(rp), dimension(:, :, :, :) :: x(nx(1),nx(2),3,trainbs)
    real(rp), dimension(:, :, :, :) :: y(ny(1),ny(2),3,trainbs)
#if defined(_OPENACC)
    attributes(device) :: y
#endif

    !$acc kernels default(present) async
    x(:, lb:ub, :, :) = y(:, :, :, :)
    !$acc end kernels
  end subroutine reshaped_copy

  subroutine distribute_batches(x_local, x, n, ng)
    use mpi
    use mod_common_mpi     , only: myid,ierr
    use mod_param, only: nranks, trainbs
    implicit none
    integer :: n(3), ng(3)
    real(rp), dimension(:, :, :, :) :: x_local(n(1),n(2),3,trainbs*nranks)
    real(rp), dimension(:, :, :, :) :: x(ng(1),ng(2),3,trainbs)

    integer :: i, offset

    if (nranks == 1) then
      !$acc kernels default(present)
      x(:,:,:,:) = x_local(:,:,:,:)
      !$acc end kernels
    else
      if (n(1) /= ng(1)) then
        print*, "ERROR: implementation only supports 1xN slabs currently..."
        stop
      endif
      if (.not. allocated(splits)) then
        allocate(splits(nranks))
        allocate(sendsizes(nranks))
        allocate(recvsizes(nranks))
        allocate(senddisps(nranks+1))
        allocate(recvdisps(nranks+1))
        allocate(dl_work(ng(1)*ng(2)*3*trainbs))

        ! Gather n2 splits
        call MPI_Allgather(n(2), 1, MPI_INTEGER, splits, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)

        ! Compute sizes and displacements
        do i = 1, nranks
          sendsizes(i) = n(1)*n(2)*3*trainbs
          recvsizes(i) = n(1)*splits(i)*3*trainbs
        enddo
        senddisps(1) = 0
        recvdisps(1) = 0
        do i = 1, nranks-1
          senddisps(i+1) = senddisps(i) + sendsizes(i)
          recvdisps(i+1) = recvdisps(i) + recvsizes(i)
        enddo
        senddisps(nranks+1) = n(1)*n(2)*3*nranks*trainbs
        recvdisps(nranks+1) = ng(1)*ng(2)*3*trainbs

      endif

      ! Distribute data
      !$acc host_data use_device(x_local)
      call MPI_Alltoallv(x_local, sendsizes, senddisps, MPI_REAL_RP, &
                         dl_work, recvsizes, recvdisps, MPI_REAL_RP, MPI_COMM_WORLD, ierr)
      !$acc end host_data

      ! Copy data into x
      !$acc wait
      offset = 1
      do i = 1, nranks
        call reshaped_copy(x, [ng(1), ng(2)], dl_work(recvdisps(i)+1), [ng(1), splits(i)], offset, offset+splits(i)-1)
        offset = offset + splits(i)
      enddo
      !$acc wait

    endif

  end subroutine distribute_batches
end module mod_dlutils
