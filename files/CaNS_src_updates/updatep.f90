! -
!
! SPDX-FileCopyrightText: Copyright (c) 2017-2022 Pedro Costa and the CaNS contributors. All rights reserved.
! SPDX-License-Identifier: MIT
!
! -

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

module mod_updatep
  use mod_types
  use mpi
  use mod_common_mpi, only: myid, ierr
  implicit none
  private
  public updatep
  contains
  subroutine updatep(n,dli,dzci,dzfi,alpha,pp,p)
    !
    ! updates the final pressure
    !
    implicit none
    integer , intent(in   ), dimension(3) :: n
    real(rp), intent(in   ), dimension(3 ) :: dli
    real(rp), intent(in   ), dimension(0:) :: dzci,dzfi
    real(rp), intent(in   ), dimension(0:,0:,0:) :: pp
    real(rp), intent(inout), dimension(0:,0:,0:) :: p
    real(rp), intent(in   ) :: alpha
    real(rp) :: dxi,dyi
    real(rp) :: pmed
    integer :: i,j,k
    !
#if defined(_IMPDIFF)
    dxi = dli(1); dyi = dli(2)
    !$acc parallel loop collapse(3) default(present) async(1)
    !$OMP PARALLEL DO   COLLAPSE(3) DEFAULT(shared)
    do k=1,n(3)
      do j=1,n(2)
        do i=1,n(1)
          p(i,j,k) = p(i,j,k) + pp(i,j,k) + alpha*( &
#if !defined(_IMPDIFF_1D)
                      (pp(i+1,j,k)-2.*pp(i,j,k)+pp(i-1,j,k))*(dxi**2) + &
                      (pp(i,j+1,k)-2.*pp(i,j,k)+pp(i,j-1,k))*(dyi**2) + &
#endif
                      ((pp(i,j,k+1)-pp(i,j,k  ))*dzci(k  ) - &
                       (pp(i,j,k  )-pp(i,j,k-1))*dzci(k-1))*dzfi(k) )
        end do
      end do
    end do
#else
    !$acc kernels default(present) async(1)
    !$OMP PARALLEL WORKSHARE
    p(1:n(1),1:n(2),1:n(3)) = p(1:n(1),1:n(2),1:n(3)) + pp(1:n(1),1:n(2),1:n(3))
    !$OMP END PARALLEL WORKSHARE
    !$acc end kernels
#endif

    ! JOSHR: fixing pressure
    !$acc wait
    if (myid == 0) then
      !$acc update host(p(1,1,n(3)/2))
      pmed = p(1,1,n(3)/2)
    endif

    call MPI_Bcast(pmed, 1, MPI_REAL_RP, 0, MPI_COMM_WORLD, ierr)
    
    !$acc parallel loop collapse(3) default(present)
    do k=0,n(3)+1
      do j=0,n(2)+1
        do i=0,n(1)+1
          p(i,j,k) = p(i,j,k) - pmed
        end do
      end do
    end do
    !$acc wait

  end subroutine updatep
end module mod_updatep
