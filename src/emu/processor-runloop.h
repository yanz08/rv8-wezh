//
//  processor-runloop.h
//

#ifndef rv_processor_runloop_h
#define rv_processor_runloop_h

namespace riscv
{

	/* Simple processor stepper with instruction cache */
	

	struct processor_singleton
	{
		static processor_singleton *current;
	};

	processor_singleton *processor_singleton::current = nullptr;

	template <typename P>
	struct processor_runloop : processor_singleton, P
	{
		static const size_t inst_cache_size = 8191;
		static const int inst_step = 100000;

		std::shared_ptr<debug_cli<P>> cli;

		struct rv_inst_cache_ent
		{
			inst_t inst;
			typename P::decode_type dec;
		};

		rv_inst_cache_ent inst_cache[inst_cache_size];

		processor_runloop() : cli(std::make_shared<debug_cli<P>>()), inst_cache() {}
		processor_runloop(std::shared_ptr<debug_cli<P>> cli) : cli(cli), inst_cache() {}

		static void signal_handler(int signum, siginfo_t *info, void *)
		{
			static_cast<processor_runloop<P> *>(processor_singleton::current)->signal_dispatch(signum, info);
		}

		void signal_dispatch(int signum, siginfo_t *info)
		{
			printf("SIGNAL   :%s pc:0x%0llx si_addr:0x%0llx\n",
				   signal_name(signum), (addr_t)P::pc, (addr_t)info->si_addr);

			/* let the processor longjmp */
			P::signal(signum, info);
		}

		void init()
		{
			// block signals before so we don't deadlock in signal handlers
			sigset_t set;
			sigemptyset(&set);
			sigaddset(&set, SIGSEGV);
			sigaddset(&set, SIGTERM);
			sigaddset(&set, SIGQUIT);
			sigaddset(&set, SIGINT);
			sigaddset(&set, SIGHUP);
			sigaddset(&set, SIGUSR1);
			if (pthread_sigmask(SIG_BLOCK, &set, NULL) != 0)
			{
				panic("can't set thread signal mask: %s", strerror(errno));
			}

			// disable unwanted signals
			sigset_t sigpipe_set;
			sigemptyset(&sigpipe_set);
			sigaddset(&sigpipe_set, SIGPIPE);
			sigprocmask(SIG_BLOCK, &sigpipe_set, nullptr);

			// install signal handler
			struct sigaction sigaction_handler;
			memset(&sigaction_handler, 0, sizeof(sigaction_handler));
			sigaction_handler.sa_sigaction = &processor_runloop<P>::signal_handler;
			sigaction_handler.sa_flags = SA_SIGINFO;
			sigaction(SIGSEGV, &sigaction_handler, nullptr);
			sigaction(SIGTERM, &sigaction_handler, nullptr);
			sigaction(SIGQUIT, &sigaction_handler, nullptr);
			sigaction(SIGINT, &sigaction_handler, nullptr);
			sigaction(SIGHUP, &sigaction_handler, nullptr);
			sigaction(SIGUSR1, &sigaction_handler, nullptr);
			processor_singleton::current = this;

			/* unblock signals */
			if (pthread_sigmask(SIG_UNBLOCK, &set, NULL) != 0)
			{
				panic("can't set thread signal mask: %s", strerror(errno));
			}

			/* processor initialization */
			P::init();
		}

		void run(exit_cause ex = exit_cause_continue)
		{
			u32 logsave = P::log;
			size_t count = inst_step;
			
			for (;;)
			{
				switch (ex)
				{
				case exit_cause_continue:
					break;
				case exit_cause_cli:
					P::debugging = true;
					count = cli->run(this);
					if (count == size_t(-1))
					{
						P::debugging = false;
						P::log = logsave;
						count = inst_step;
					}
					else
					{
						P::log |= (proc_log_inst | proc_log_operands | proc_log_trap);
					}
					break;
				case exit_cause_poweroff:
					return;
				}
				ex = step_count(count);
				if (P::debugging && ex == exit_cause_continue)
				{
					ex = exit_cause_cli;
				}	
			}
		
		}

		exit_cause step_count(size_t count)
		{
			typename P::decode_type dec;
			typename P::ux inststop = P::instret + count;
			typename P::ux pc_offset, new_offset;
			inst_t inst = 0, inst_cache_key;
			

			/* interrupt service routine */
			P::time = cpu_cycle_clock();
			P::isr();

			/* trap return path */
			int cause;
			if (unlikely((cause = setjmp(P::env)) > 0))
			{
				cause -= P::internal_cause_offset;
				switch (cause)
				{
				case P::internal_cause_cli:
					return exit_cause_cli;
				case P::internal_cause_fatal:
					P::print_csr_registers();
					P::print_int_registers();
					return exit_cause_poweroff;
				case P::internal_cause_poweroff:
					return exit_cause_poweroff;
				}
				P::trap(dec, cause);
				if (!P::running)
					return exit_cause_poweroff;
			}

			/* step the processor */
			while (P::instret != inststop)
			{
				if (P::pc == P::breakpoint && P::breakpoint != 0)
				{
					return exit_cause_cli;
				}
				inst = P::mmu.inst_fetch(*this, P::pc, pc_offset);
				inst_cache_key = inst % inst_cache_size;
				if (inst_cache[inst_cache_key].inst == inst)
				{
					dec = inst_cache[inst_cache_key].dec;
				}
				else
				{
					P::inst_decode(dec, inst);
					inst_cache[inst_cache_key].inst = inst;
					inst_cache[inst_cache_key].dec = dec;
				}
				if ((new_offset = P::inst_exec(dec, pc_offset)) != typename P::ux(-1) ||
					(new_offset = P::inst_priv(dec, pc_offset)) != typename P::ux(-1))
				{
					//if (P::log) P::print_log(dec, inst);
					if (P::log)
					{
						P::print_log(dec, inst);
						std::string exe_ins = P::print_log_opcode(dec, inst);
						//printf("------------%08llx : %s \n", addr_t(P::pc), exe_ins.c_str());
						if (exe_ins == "addi")
						{
							P::num_ADDI++;
							P::I_type++;
							//printf("------------%08llx : %s %lld \n", addr_t(P::pc), exe_ins.c_str(), P::num_ADDI);
						} else if (exe_ins == "slti")
						{
							P::num_SLTI++;
							P::I_type++;
						} else if (exe_ins == "sltiu")
						{
							P::num_SLTIU++;
							P::I_type++;
						} else if (exe_ins == "xori")
						{
							P::num_XORI++;
							P::I_type++;
						} else if (exe_ins == "ori")
						{
							P::num_ORI++;
							P::I_type++;
						} else if (exe_ins == "andi")
						{
							P::num_ANDI++;
							P::I_type++;
						} else if (exe_ins == "add")
						{
							P::num_ADD++;
							P::R_type++;
						} else if (exe_ins == "sub")
						{
							P::num_SUB++;
							P::R_type++;
						} else if (exe_ins == "sll")
						{
							P::num_SLL++;
							P::R_type++;
						} else if (exe_ins == "slt")
						{
							P::num_SLT++;
							P::R_type++;
						} else if (exe_ins == "sltu")
						{
							P::num_SLTU++;
							P::R_type++;
						} else if (exe_ins == "srl") {
							P::num_SRL += 1;
							P::R_type += 1;
						} else if (exe_ins == "sra") {
							P::num_SRA += 1;
							P::R_type += 1;
						} else if (exe_ins == "xor") {
							P::num_XOR += 1;
							P::R_type += 1;
						} else if (exe_ins == "or") {
							P::num_OR += 1;
							P::R_type += 1;
						} else if (exe_ins == "and") {
							P::num_AND += 1;
							P::R_type += 1;
						} else if (exe_ins == "slli") {
							P::num_SLLI += 1;
							P::I_type += 1;
						} else if (exe_ins == "srli") {
							P::num_SRLI += 1;
							P::I_type += 1;
						} else if (exe_ins == "srai") {
							P::num_SRAI += 1;
							P::I_type += 1;
						} else if (exe_ins == "lui") {
							P::num_LUI += 1;
							P::U_type += 1;
						} else if (exe_ins == "auipc") {
							P::num_AUIPC += 1;
							P::U_type += 1;
						} else if (exe_ins == "jal") {
							P::num_JAL += 1;
							P::J_type += 1;
						} else if (exe_ins == "jalr") {
							P::num_JALR += 1;
							P::I_type += 1;
						} else if (exe_ins == "sb") {
							P::num_SB += 1;
							P::S_type += 1;
						} else if (exe_ins == "sh") {
							P::num_SH += 1;
							P::S_type += 1;
						} else if (exe_ins == "sw") {
							P::num_SW += 1;
							P::S_type += 1;
						} else if (exe_ins == "lb") {
							P::num_LB += 1;
							P::I_type += 1;
						} else if (exe_ins == "lh") {
							P::num_LH += 1;
							P::I_type += 1;
						} else if (exe_ins == "lw") {
							P::num_LW += 1;
							P::I_type += 1;
						} else if (exe_ins == "lbu") {
							P::num_LBU += 1;
							P::I_type += 1;
						} else if (exe_ins == "lhu") {
							P::num_LHU += 1;
							P::I_type += 1;
						} else if (exe_ins == "beq") {
							P::num_BEQ += 1;
							P::B_type += 1;
						} else if (exe_ins == "bne") {
							P::num_BNE += 1;
							P::B_type += 1;
						} else if (exe_ins == "blt") {
							P::num_BLT += 1;
							P::B_type += 1;
						} else if (exe_ins == "bge") {
							P::num_BGE += 1;
							P::B_type += 1;
						} else if (exe_ins == "bltu") {
							P::num_BLTU += 1;
							P::B_type += 1;
						} else if (exe_ins == "bgeu") {
							P::num_BGEU += 1;
							P::B_type += 1;
						} else if (exe_ins == "fence") {
							P::num_FENCE += 1;
							P::I_type += 1;
						} else if (exe_ins == "ecall") {
							P::num_ECALL += 1;
							P::I_type += 1;
						} else if (exe_ins == "ebreak") {
							P::num_EBREAK += 1;
							P::I_type += 1;
						} else if (exe_ins == "fence.i") { //Zifencei standard extension
							P::num_FENCE_I += 1;
							P::I_type += 1;
						} else if (exe_ins == "csrrw") { // Zicsr standard extension
							P::num_CSRRW += 1;
							P::I_type += 1;
						} else if (exe_ins == "csrrs") {
							P::num_CSRRS += 1;
							P::I_type += 1;
						} else if (exe_ins == "csrrc") {
							P::num_CSRRC += 1;
							P::I_type += 1;
						} else if (exe_ins == "csrrwi") {
							P::num_CSRRWI += 1;
							P::I_type += 1;
						} else if (exe_ins == "csrrsi") {
							P::num_CSRRSI += 1;
							P::I_type += 1;
						} else if (exe_ins == "csrrci") {
							P::num_CSRRCI += 1;
							P::I_type += 1;
						} else if (exe_ins == "mul") { // RV32M standard extension
							P::num_MUL += 1;
							P::R_type += 1;
						} else if (exe_ins == "mulh") {
							P::num_MULH += 1;
							P::R_type += 1;
						} else if (exe_ins == "mulhsu") {
							P::num_MULHSU += 1;
							P::R_type += 1;
						} else if (exe_ins == "mulhu") {
							P::num_MULHU += 1;
							P::R_type += 1;
						} else if (exe_ins == "div") {
							P::num_DIV += 1;
							P::R_type += 1;
						} else if (exe_ins == "divu") {
							P::num_DIVU += 1;
							P::R_type += 1;
						} else if (exe_ins == "rem") {
							P::num_REM += 1;
							P::R_type += 1;
						} else if (exe_ins == "remu") {
							P::num_REMU += 1;
							P::R_type += 1;
						} else if (exe_ins == "lr.w") { // RV32A standard extension
							P::num_LR_W += 1;
							P::R_type += 1;
						} else if (exe_ins == "sc.w") {
							P::num_SC_W += 1;
							P::R_type += 1;
						} else if (exe_ins == "amoswap.w") {
							P::num_AMOSWAP_W += 1;
							P::R_type += 1;
						} else if (exe_ins == "amoadd.w") {
							P::num_AMOADD_W += 1;
							P::R_type += 1;
						} else if (exe_ins == "amoxor.w") {
							P::num_AMOXOR_W += 1;
							P::R_type += 1;
						} else if (exe_ins == "amoand.w") {
							P::num_AMOAND_W += 1;
							P::R_type += 1;
						} else if (exe_ins == "amoor.w") {
							P::num_AMOOR_W += 1;
							P::R_type += 1;
						} else if (exe_ins == "amomin.w") {
							P::num_AMOMIN_W += 1;
							P::R_type += 1;
						} else if (exe_ins == "amomax.w") {
							P::num_AMOMAX_W += 1;
							P::R_type += 1;
						} else if (exe_ins == "amominu.w") {
							P::num_AMOMINU_W += 1;
							P::R_type += 1;
						} else if (exe_ins == "amomaxu.w") {
							P::num_AMOMAXU_W += 1;
							P::R_type += 1;
						} else if (exe_ins == "uret") { // privileged instructions
							P::num_URET += 1;
							P::I_type += 1;
						} else if (exe_ins == "sret") {
							P::num_SRET += 1;
							P::I_type += 1;
						} else if (exe_ins == "mret") {
							P::num_MRET += 1;
							P::I_type += 1;
						} else if (exe_ins == "wfi") {
							P::num_WFI += 1;
							P::I_type += 1;
						} else if (exe_ins == "sfence.vma") {
							P::num_SFENCE_VMA += 1;
							P::R_type += 1;
						}
						
					}
					P::pc += new_offset;
					P::instret++;
				} 
				else
				{
					P::raise(rv_cause_illegal_instruction, P::pc);
				}
			}
			return exit_cause_continue;
		}
//--------------------------------------------------
		exit_cause step(size_t count)
		{
			typename P::decode_type dec;
			typename P::ux inststop = P::instret + count;
			typename P::ux pc_offset, new_offset;
			inst_t inst = 0, inst_cache_key;
			

			/* interrupt service routine */
			P::time = cpu_cycle_clock();
			P::isr();

			/* trap return path */
			int cause;
			if (unlikely((cause = setjmp(P::env)) > 0))
			{
				cause -= P::internal_cause_offset;
				switch (cause)
				{
				case P::internal_cause_cli:
					return exit_cause_cli;
				case P::internal_cause_fatal:
					P::print_csr_registers();
					P::print_int_registers();
					return exit_cause_poweroff;
				case P::internal_cause_poweroff:
					return exit_cause_poweroff;
				}
				P::trap(dec, cause);
				if (!P::running)
					return exit_cause_poweroff;
			}

			/* step the processor */
			while (P::instret != inststop)
			{
				if (P::pc == P::breakpoint && P::breakpoint != 0)
				{
					return exit_cause_cli;
				}
				inst = P::mmu.inst_fetch(*this, P::pc, pc_offset);
				inst_cache_key = inst % inst_cache_size;
				if (inst_cache[inst_cache_key].inst == inst)
				{
					dec = inst_cache[inst_cache_key].dec;
				}
				else
				{
					P::inst_decode(dec, inst);
					inst_cache[inst_cache_key].inst = inst;
					inst_cache[inst_cache_key].dec = dec;
				}
				if ((new_offset = P::inst_exec(dec, pc_offset)) != typename P::ux(-1) ||
					(new_offset = P::inst_priv(dec, pc_offset)) != typename P::ux(-1))
				{
					if (P::log) P::print_log(dec, inst);
					P::pc += new_offset;
					P::instret++;
				}
				else
				{
					P::raise(rv_cause_illegal_instruction, P::pc);
				}
			}
			return exit_cause_continue;
		}
	};

}

#endif
